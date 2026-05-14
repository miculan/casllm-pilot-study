"""Smoke tests for the Blocksworld validator."""

from __future__ import annotations
import json
from pathlib import Path

from validator import validate, extract_plan

ROOT = Path(__file__).resolve().parent
TASKS = ROOT / "tasks"


def load(task_id):
    return json.loads((TASKS / f"{task_id}.json").read_text())


# A handful of hand-coded reference solutions
REFERENCE = {
    "bw-02": """PLAN:
1. pickup(b)
2. stack(b, c)
3. pickup(a)
4. stack(a, b)
""",
    "bw-04": """PLAN:
1. unstack(c, b)
2. putdown(c)
3. unstack(b, a)
4. putdown(b)
""",
    "bw-05": """PLAN:
1. unstack(c, b)
2. putdown(c)
3. pickup(b)
4. stack(b, a)
5. pickup(c)
6. stack(c, b)
""",
    "bw-11": """PLAN:
1. unstack(b, a)
2. putdown(b)
3. unstack(d, c)
4. stack(d, b)
5. pickup(c)
6. stack(c, d)
7. pickup(a)
8. wait
""",  # Intentionally wrong to test failure detection (incomplete goal + bad action)
    "bw-13": """PLAN:
1. pickup(c)
2. stack(c, d)
3. pickup(b)
4. stack(b, c)
5. pickup(a)
6. stack(a, b)
""",
}


def main():
    print("== smoke test: extract_plan ==")
    p = extract_plan("Sure, here is the plan.\n\nPLAN:\n1. pickup(a)\n2. stack(a, b)\n\nDone!")
    assert p == ["pickup(a)", "stack(a, b)"], p
    print("  preamble + trailing text: OK")

    # No PLAN: header but numbered actions
    p = extract_plan("1. pickup(a)\n2. putdown(a)")
    assert p == ["pickup(a)", "putdown(a)"], p
    print("  no header, just numbered actions: OK")

    # Bullets instead of numbers
    p = extract_plan("PLAN:\n- pickup(a)\n- putdown(a)")
    assert p == ["pickup(a)", "putdown(a)"], p
    print("  bullet actions: OK")

    # With code fence-style backticks
    p = extract_plan("PLAN:\n1. `pickup(a)`\n2. `putdown(a)`")
    assert p == ["pickup(a)", "putdown(a)"], p
    print("  backticked actions: OK")

    print()
    print("== smoke test: validate ==")
    for tid, plan in REFERENCE.items():
        task = load(tid)
        r = validate(task, plan)
        outcome = "PASS" if r.success else "FAIL"
        print(f"  {tid}: {outcome:4s}  ({r.n_actions} actions)  {r.reason}")

    print()
    print("== adversarial cases ==")
    # Empty response
    r = validate(load("bw-02"), "I cannot solve this.")
    assert not r.success
    print(f"  empty plan: FAIL  ({r.reason})")

    # Wrong syntax
    r = validate(load("bw-02"), "PLAN:\n1. move b on c\n2. move a on b")
    assert not r.success
    print(f"  natural-language actions: FAIL  ({r.reason})")

    # Precondition violation
    r = validate(load("bw-02"), "PLAN:\n1. stack(b, c)\n2. stack(a, b)")
    assert not r.success
    print(f"  stack without holding: FAIL  ({r.reason})")

    # Reference for bw-02 should pass (already tested but explicit)
    r = validate(load("bw-02"), REFERENCE["bw-02"])
    assert r.success, r.reason
    print(f"  bw-02 reference: PASS")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
