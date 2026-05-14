"""
Blocksworld plan validator for the CASeLLM pilot study.

Given a task JSON (initial state + goal) and a candidate plan (text), simulate
the plan and decide whether it correctly achieves the goal. Returns a
structured Result with success/failure and a reason.

Usage:
    python validator.py tasks/bw-01.json answers/claude/bw-01.txt
    python validator.py --batch answers/claude/   # validates all answers in dir
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# State representation
# ---------------------------------------------------------------------------

@dataclass
class State:
    """Mutable Blocksworld state."""
    on_table: set[str] = field(default_factory=set)   # blocks directly on table
    on: dict[str, str] = field(default_factory=dict)  # on[x] = y  iff x on y
    holding: str | None = None                        # block in the arm or None

    @property
    def handempty(self) -> bool:
        return self.holding is None

    @property
    def blocks(self) -> set[str]:
        all_b = set(self.on_table) | set(self.on.keys()) | set(self.on.values())
        if self.holding is not None:
            all_b.add(self.holding)
        return all_b

    def is_clear(self, b: str) -> bool:
        """A block is clear iff nothing is on top of it and it is not being held."""
        if self.holding == b:
            return False
        return b not in self.on.values()

    def copy(self) -> "State":
        return State(set(self.on_table), dict(self.on), self.holding)


def state_from_json(spec: dict) -> State:
    """Construct a State from a task JSON's 'initial' or 'goal' section."""
    return State(
        on_table=set(spec.get("on_table", [])),
        on=dict(spec.get("on", {})),
        holding=None,
    )


# ---------------------------------------------------------------------------
# Action semantics
# ---------------------------------------------------------------------------

ACTION_RE = re.compile(
    r"^\s*"
    r"(?P<verb>pickup|putdown|stack|unstack)"
    r"\s*\(\s*"
    r"(?P<arg1>[a-zA-Z][a-zA-Z0-9_]*)"
    r"(?:\s*,\s*(?P<arg2>[a-zA-Z][a-zA-Z0-9_]*))?"
    r"\s*\)\s*$",
    re.IGNORECASE,
)


@dataclass
class ActionError(Exception):
    step: int
    action: str
    reason: str

    def __str__(self):
        return f"step {self.step}: '{self.action}' invalid — {self.reason}"


def apply_action(s: State, action: str, step: int) -> None:
    """Apply `action` to state `s` in place. Raises ActionError on bad action."""
    m = ACTION_RE.match(action)
    if not m:
        raise ActionError(step, action, "malformed (does not match action syntax)")
    verb = m["verb"].lower()
    x = m["arg1"]
    y = m["arg2"]

    if verb in ("pickup", "putdown") and y is not None:
        raise ActionError(step, action, f"{verb} takes 1 argument, got 2")
    if verb in ("stack", "unstack") and y is None:
        raise ActionError(step, action, f"{verb} takes 2 arguments, got 1")

    blocks = s.blocks
    if x not in blocks:
        raise ActionError(step, action, f"unknown block {x!r}")
    if y is not None and y not in blocks:
        raise ActionError(step, action, f"unknown block {y!r}")
    if y is not None and x == y:
        raise ActionError(step, action, f"cannot reference same block twice ({x})")

    if verb == "pickup":
        if not s.handempty:
            raise ActionError(step, action, f"arm is holding {s.holding}, not empty")
        if x not in s.on_table:
            raise ActionError(step, action, f"{x} is not on the table")
        if not s.is_clear(x):
            raise ActionError(step, action, f"{x} is not clear")
        s.on_table.remove(x)
        s.holding = x

    elif verb == "putdown":
        if s.holding != x:
            raise ActionError(step, action,
                              f"arm is holding {s.holding!r}, not {x!r}")
        s.on_table.add(x)
        s.holding = None

    elif verb == "stack":
        if s.holding != x:
            raise ActionError(step, action,
                              f"arm is holding {s.holding!r}, not {x!r}")
        if not s.is_clear(y):
            raise ActionError(step, action, f"{y} is not clear")
        s.on[x] = y
        s.holding = None

    elif verb == "unstack":
        if not s.handempty:
            raise ActionError(step, action, f"arm is holding {s.holding}, not empty")
        if s.on.get(x) != y:
            raise ActionError(step, action, f"{x} is not on {y}")
        if not s.is_clear(x):
            raise ActionError(step, action, f"{x} is not clear")
        del s.on[x]
        s.holding = x


def goal_satisfied(s: State, goal_on_table: set[str], goal_on: dict[str, str]) -> tuple[bool, list[str]]:
    """Check whether state s satisfies the goal. Returns (ok, list of unmet facts)."""
    unmet = []
    for b in goal_on_table:
        if b not in s.on_table:
            unmet.append(f"{b} should be on the table")
    for x, y in goal_on.items():
        if s.on.get(x) != y:
            unmet.append(f"{x} should be on {y}")
    if s.holding is not None:
        unmet.append(f"arm should be empty at the end, but holds {s.holding}")
    return (len(unmet) == 0, unmet)


# ---------------------------------------------------------------------------
# Plan extraction
# ---------------------------------------------------------------------------

# Match a numbered or bulleted line containing a Blocksworld action.
PLAN_LINE_RE = re.compile(
    r"^\s*(?:\d+\.|\d+\)|-|\*)\s*"   # optional leading number/dot/dash
    r"(?P<body>.*?)"
    r"\s*$",
)


def extract_plan(text: str) -> list[str]:
    """Extract the list of action strings from an LLM response.

    Strategy:
      1. Find the last occurrence of 'PLAN:' (case-insensitive) and start there.
         If absent, scan the whole text.
      2. From there on, collect lines that look like numbered/bulleted actions.
      3. Stop at the first blank line *after* we have collected at least one
         action, OR at end of text.
    """
    # Locate PLAN: header
    upper = text.upper()
    idx = upper.rfind("PLAN:")
    if idx >= 0:
        body = text[idx + len("PLAN:"):]
    else:
        body = text

    actions = []
    started = False
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            if started:
                break
            else:
                continue
        m = PLAN_LINE_RE.match(line)
        if not m:
            continue
        cand = m["body"].strip()
        # Strip optional markdown/backticks
        cand = cand.strip("`").strip()
        # The candidate must match the action regex to count
        if ACTION_RE.match(cand):
            actions.append(cand)
            started = True
        elif started:
            # We were collecting actions and hit a non-action line — stop
            break
    return actions


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    task_id: str
    success: bool
    reason: str
    n_actions: int
    extracted_plan: list[str]
    failure_step: int | None = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "reason": self.reason,
            "n_actions": self.n_actions,
            "failure_step": self.failure_step,
            "extracted_plan": self.extracted_plan,
        }


def validate(task: dict, response_text: str) -> ValidationResult:
    plan = extract_plan(response_text)
    if not plan:
        return ValidationResult(
            task_id=task["id"],
            success=False,
            reason="no plan extracted from response",
            n_actions=0,
            extracted_plan=[],
        )

    state = state_from_json(task["initial"])
    goal_on_table = set(task["goal"]["on_table"])
    goal_on = dict(task["goal"]["on"])

    for i, action in enumerate(plan, start=1):
        try:
            apply_action(state, action, i)
        except ActionError as e:
            return ValidationResult(
                task_id=task["id"],
                success=False,
                reason=str(e),
                n_actions=len(plan),
                extracted_plan=plan,
                failure_step=i,
            )

    ok, unmet = goal_satisfied(state, goal_on_table, goal_on)
    if not ok:
        return ValidationResult(
            task_id=task["id"],
            success=False,
            reason="plan executed but goal not satisfied: " + "; ".join(unmet),
            n_actions=len(plan),
            extracted_plan=plan,
        )
    return ValidationResult(
        task_id=task["id"],
        success=True,
        reason="goal achieved",
        n_actions=len(plan),
        extracted_plan=plan,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_task(task_path: Path) -> dict:
    return json.loads(task_path.read_text())


def _resolve_task_for_answer(answer_path: Path, tasks_dir: Path) -> Path:
    """An answer file is named bw-XX.txt; find tasks_dir/bw-XX.json."""
    stem = answer_path.stem  # bw-XX
    return tasks_dir / f"{stem}.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task_or_dir", help="Path to task .json OR (with --batch) an answer dir")
    p.add_argument("answer", nargs="?", help="Path to answer .txt (single-task mode)")
    p.add_argument("--batch", action="store_true",
                   help="Treat first arg as an answer directory and validate everything")
    p.add_argument("--tasks-dir", default=None,
                   help="Tasks directory (default: ../tasks relative to answer dir)")
    args = p.parse_args()

    if args.batch:
        ans_dir = Path(args.task_or_dir)
        tasks_dir = Path(args.tasks_dir) if args.tasks_dir else ans_dir.parent.parent / "tasks"
        results = []
        for ans_file in sorted(ans_dir.glob("bw-*.txt")):
            task_path = _resolve_task_for_answer(ans_file, tasks_dir)
            if not task_path.exists():
                print(f"skip {ans_file.name}: no task file")
                continue
            task = _load_task(task_path)
            response = ans_file.read_text()
            r = validate(task, response)
            results.append(r)
            mark = "PASS" if r.success else "FAIL"
            print(f"  {r.task_id}  {mark:4s}  ({r.n_actions} actions)  {r.reason}")
        n = len(results)
        n_pass = sum(1 for r in results if r.success)
        print(f"\nSummary: {n_pass}/{n} passed  ({n_pass/n:.1%})" if n else "no answers found")
        # Write a summary JSON
        out = ans_dir / "summary.json"
        out.write_text(json.dumps([r.to_dict() for r in results], indent=2))
        print(f"Wrote {out}")
    else:
        task = _load_task(Path(args.task_or_dir))
        response = Path(args.answer).read_text()
        r = validate(task, response)
        print(json.dumps(r.to_dict(), indent=2))
        sys.exit(0 if r.success else 1)


if __name__ == "__main__":
    main()
