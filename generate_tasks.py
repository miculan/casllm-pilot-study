"""
Generate 30 Blocksworld planning tasks for the CASeLLM pilot study.
10 easy (3 blocks), 10 medium (4-5 blocks), 10 hard (6-8 blocks).

Each task is saved in two formats:
  - tasks/bw-XX.txt : human/LLM-readable natural-language description
  - tasks/bw-XX.json: machine-readable initial state + goal for the validator
"""

from __future__ import annotations
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS_DIR = ROOT / "tasks"
TASKS_DIR.mkdir(exist_ok=True, parents=True)


def state_to_facts(stacks: list[list[str]]) -> dict:
    """Convert a list of stacks (each stack bottom-to-top) into a fact dict.

    Returns dict with:
      on_table: set of block names on the table
      on:       dict {x: y} meaning block x is directly on block y
      clear:    set of block names with nothing on top
    """
    on_table = set()
    on = {}
    all_blocks = set()
    for stack in stacks:
        for b in stack:
            all_blocks.add(b)
        if stack:
            on_table.add(stack[0])
            for i in range(1, len(stack)):
                on[stack[i]] = stack[i - 1]
    # Compute clear: a block is clear if nothing is on top of it
    covered = set(on.values())
    clear = all_blocks - covered
    return {
        "blocks": sorted(all_blocks),
        "on_table": sorted(on_table),
        "on": on,
        "clear": sorted(clear),
        "handempty": True,
        "holding": None,
    }


def render_state_text(state: dict) -> str:
    """Render a state as bullet-list natural language."""
    lines = []
    for b in state["on_table"]:
        lines.append(f"- Block {b} is on the table.")
    for x, y in sorted(state["on"].items()):
        lines.append(f"- Block {x} is on top of block {y}.")
    for b in state["clear"]:
        lines.append(f"- Block {b} is clear (nothing on top).")
    lines.append("- The robot arm is empty.")
    return "\n".join(lines)


def render_goal_text(state: dict) -> str:
    """Render only the *positional* facts of the goal (no clear, no handempty)."""
    lines = []
    for b in state["on_table"]:
        lines.append(f"- Block {b} is on the table.")
    for x, y in sorted(state["on"].items()):
        lines.append(f"- Block {x} is on top of block {y}.")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Task definitions
# -----------------------------------------------------------------------------
# Each task is (id, difficulty, initial_stacks, goal_stacks, reference_length).
# Stacks are listed bottom-to-top. All loose blocks must be in their own
# singleton list. The reference_length is the optimal plan length (used only
# for our own bookkeeping, NOT shown to the LLM).
# -----------------------------------------------------------------------------

TASKS = [
    # ---------------- EASY: 3 blocks ----------------
    ("bw-01", "easy",
     [["a", "b", "c"]],
     [["c", "b", "a"]],
     6),
    ("bw-02", "easy",
     [["a"], ["b"], ["c"]],
     [["c", "b", "a"]],
     4),
    ("bw-03", "easy",
     [["a", "b"], ["c"]],
     [["c", "a"], ["b"]],
     4),
    ("bw-04", "easy",
     [["a", "b", "c"]],
     [["a"], ["b"], ["c"]],
     4),
    ("bw-05", "easy",
     [["a"], ["b", "c"]],
     [["a", "b", "c"]],
     2),
    ("bw-06", "easy",
     [["c", "a", "b"]],
     [["a", "b", "c"]],
     6),
    ("bw-07", "easy",
     [["a", "c"], ["b"]],
     [["b", "a", "c"]],
     4),
    ("bw-08", "easy",
     [["b", "a"], ["c"]],
     [["a", "c", "b"]],
     5),
    ("bw-09", "easy",
     [["a"], ["c", "b"]],
     [["b"], ["a", "c"]],
     4),
    ("bw-10", "easy",
     [["b", "c", "a"]],
     [["a", "b", "c"]],
     6),

    # ---------------- MEDIUM: 4-5 blocks ----------------
    ("bw-11", "medium",
     [["a", "b"], ["c", "d"]],
     [["a", "b", "c", "d"]],
     4),
    ("bw-12", "medium",
     [["a", "b", "c", "d"]],
     [["d", "c", "b", "a"]],
     12),
    ("bw-13", "medium",
     [["a"], ["b"], ["c"], ["d"]],
     [["d", "c", "b", "a"]],
     6),
    ("bw-14", "medium",
     [["a", "c"], ["b", "d"]],
     [["a", "b"], ["c", "d"]],
     4),
    ("bw-15", "medium",
     [["c", "a"], ["d", "b"]],
     [["a", "b", "c", "d"]],
     8),
    ("bw-16", "medium",
     [["a", "b", "c"], ["d", "e"]],
     [["e", "d", "c", "b", "a"]],
     14),
    ("bw-17", "medium",
     [["a"], ["b", "c"], ["d", "e"]],
     [["a", "b", "c"], ["d", "e"]],
     2),
    ("bw-18", "medium",
     [["e", "a"], ["b"], ["c"], ["d"]],
     [["a", "b", "c", "d", "e"]],
     8),
    ("bw-19", "medium",
     [["a", "e", "b"], ["c", "d"]],
     [["b", "d"], ["a", "c", "e"]],
     10),
    ("bw-20", "medium",
     [["e", "d", "c", "b", "a"]],
     [["a", "b", "c", "d", "e"]],
     12),

    # ---------------- HARD: 6-8 blocks ----------------
    ("bw-21", "hard",
     [["a", "b", "c", "d", "e", "f"]],
     [["f", "e", "d", "c", "b", "a"]],
     20),
    ("bw-22", "hard",
     [["a"], ["b"], ["c"], ["d"], ["e"], ["f"]],
     [["f", "e", "d", "c", "b", "a"]],
     10),
    ("bw-23", "hard",
     [["a", "b"], ["c", "d"], ["e", "f"]],
     [["a", "c", "e"], ["b", "d", "f"]],
     8),
    ("bw-24", "hard",
     [["a", "b", "c"], ["d", "e", "f"]],
     [["a", "d"], ["b", "e"], ["c", "f"]],
     10),
    ("bw-25", "hard",
     [["b", "f", "a"], ["d", "c"], ["e"]],
     [["a", "b", "c", "d", "e", "f"]],
     14),
    ("bw-26", "hard",
     [["a", "b", "c", "d", "e", "f", "g"]],
     [["g", "f", "e", "d", "c", "b", "a"]],
     24),
    ("bw-27", "hard",
     [["a", "c", "e", "g"], ["b", "d", "f"]],
     [["a", "b", "c", "d", "e", "f", "g"]],
     14),
    ("bw-28", "hard",
     [["g"], ["f"], ["e"], ["d"], ["c"], ["b"], ["a"]],
     [["a", "b", "c", "d", "e", "f", "g"]],
     12),
    ("bw-29", "hard",
     [["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]],
     [["h", "g", "f", "e", "d", "c", "b", "a"]],
     22),
    ("bw-30", "hard",
     [["c", "a", "f"], ["b", "g", "d"], ["e", "h"]],
     [["a", "b", "c", "d"], ["e", "f", "g", "h"]],
     20),
]


def write_task(task_id, difficulty, init_stacks, goal_stacks, ref_len):
    init = state_to_facts(init_stacks)
    goal = state_to_facts(goal_stacks)
    # Sanity check: same block set
    assert set(init["blocks"]) == set(goal["blocks"]), (
        f"{task_id}: blocks differ between init and goal"
    )

    # --- machine-readable JSON ---
    json_blob = {
        "id": task_id,
        "difficulty": difficulty,
        "blocks": init["blocks"],
        "initial": {
            "on_table": init["on_table"],
            "on": init["on"],
        },
        "goal": {
            "on_table": goal["on_table"],
            "on": goal["on"],
        },
        "reference_optimal_length": ref_len,
    }
    (TASKS_DIR / f"{task_id}.json").write_text(json.dumps(json_blob, indent=2))

    # --- human/LLM-readable text ---
    txt = []
    txt.append(f"# Task {task_id} (Blocksworld, difficulty: {difficulty})")
    txt.append("")
    txt.append("Domain rules:")
    txt.append("- Blocks are either on the table or stacked on exactly one other block.")
    txt.append("- The robot arm can hold at most one block at a time.")
    txt.append("- Actions:")
    txt.append("    * pickup(X)    -- requires X on table, X clear, arm empty")
    txt.append("    * putdown(X)   -- requires arm holding X")
    txt.append("    * stack(X, Y)  -- requires arm holding X, Y clear")
    txt.append("    * unstack(X, Y)-- requires X on Y, X clear, arm empty")
    txt.append("")
    txt.append(f"Blocks: {', '.join(init['blocks'])}")
    txt.append("")
    txt.append("Initial state:")
    txt.append(render_state_text(init))
    txt.append("")
    txt.append("Goal state:")
    txt.append(render_goal_text(goal))
    txt.append("")
    txt.append("Output a plan that achieves the goal, using EXACTLY this format:")
    txt.append("PLAN:")
    txt.append("1. <action>")
    txt.append("2. <action>")
    txt.append("...")
    txt.append("")
    txt.append("Each <action> must be one of: pickup(X), putdown(X), stack(X, Y), unstack(X, Y).")
    txt.append("Do not output anything before 'PLAN:' or after the last numbered line.")
    (TASKS_DIR / f"{task_id}.txt").write_text("\n".join(txt))


def main():
    for t in TASKS:
        write_task(*t)
    print(f"Wrote {len(TASKS)} tasks to {TASKS_DIR}")
    # Print summary by difficulty
    by_diff = {}
    for _, d, *_ in TASKS:
        by_diff[d] = by_diff.get(d, 0) + 1
    for d, n in by_diff.items():
        print(f"  {d}: {n}")


if __name__ == "__main__":
    main()
