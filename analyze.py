"""
Aggregate pilot validator results across all models and compute:
  - per-model success/failure rates p_i
  - pairwise joint failure rates q(i,j) and the ratio q/min(p_i,p_j)
  - per-difficulty breakdowns
  - by-task table

Writes:
  results/per_model.csv
  results/pairwise.csv
  results/per_task.csv
  results/summary.txt    # human-readable
  results/summary.json   # machine-readable, ready to pull into LaTeX
"""

from __future__ import annotations
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANS = ROOT / "answers"
TASKS = ROOT / "tasks"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def load_summary(d: Path) -> dict[str, bool]:
    """Returns {task_id: success_bool}."""
    s = json.loads((d / "summary.json").read_text())
    return {r["task_id"]: r["success"] for r in s}


def load_run_meta(d: Path) -> dict:
    p = d / "run_metadata.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def task_difficulty(task_id: str) -> str:
    j = json.loads((TASKS / f"{task_id}.json").read_text())
    return j["difficulty"]


def main():
    # Discover all answer directories with a summary.json
    runs = {}
    for d in sorted(ANS.iterdir()):
        if (d / "summary.json").exists():
            runs[d.name] = {
                "results": load_summary(d),
                "meta": load_run_meta(d),
            }
    if not runs:
        print("No runs found. Run eval_*.py then validator.py --batch.")
        return

    print(f"Found {len(runs)} runs: {', '.join(runs)}")

    # All task ids (union)
    all_tids = sorted({tid for r in runs.values() for tid in r["results"]})

    # ------------------------------------------------------------------
    # Per-model stats
    # ------------------------------------------------------------------
    per_model = {}
    for name, r in runs.items():
        results = r["results"]
        n = len(results)
        n_fail = sum(1 for ok in results.values() if not ok)
        p = n_fail / n if n else 0.0
        # by-difficulty
        by_diff = defaultdict(lambda: [0, 0])  # diff -> [fail, total]
        for tid, ok in results.items():
            d = task_difficulty(tid)
            by_diff[d][1] += 1
            if not ok:
                by_diff[d][0] += 1
        per_model[name] = {
            "model": r["meta"].get("model", name),
            "n": n,
            "n_fail": n_fail,
            "p": p,
            "p_by_difficulty": {d: f / t for d, (f, t) in by_diff.items()},
            "counts_by_difficulty": {d: {"fail": f, "total": t}
                                     for d, (f, t) in by_diff.items()},
        }

    # ------------------------------------------------------------------
    # Pairwise joint failure
    # ------------------------------------------------------------------
    pairwise = {}
    names = sorted(runs)
    for a, b in combinations(names, 2):
        common = sorted(set(runs[a]["results"]) & set(runs[b]["results"]))
        n = len(common)
        joint_fail = sum(1 for tid in common
                         if not runs[a]["results"][tid]
                         and not runs[b]["results"][tid])
        p_a = per_model[a]["p"]
        p_b = per_model[b]["p"]
        q = joint_fail / n if n else 0.0
        # Under independence: q_indep = p_a * p_b
        q_indep = p_a * p_b
        # q / min(p): how concentrated is the overlap on the weaker component
        denom = min(p_a, p_b)
        q_over_min = (q / denom) if denom > 0 else 0.0
        # R_pair = 1 - q
        r_pair = 1 - q
        pairwise[(a, b)] = {
            "a": a, "b": b,
            "n": n,
            "p_a": p_a, "p_b": p_b,
            "q": q,
            "q_indep_if_uncorrelated": q_indep,
            "q_over_min_p": q_over_min,
            "R_pair": r_pair,
            "joint_failures": joint_fail,
        }

    # ------------------------------------------------------------------
    # Per-task table (rows: tasks; cols: model+difficulty+success per model)
    # ------------------------------------------------------------------
    per_task_rows = []
    for tid in all_tids:
        row = {"task": tid, "difficulty": task_difficulty(tid)}
        for name in names:
            row[name] = "P" if runs[name]["results"].get(tid) else "F"
        per_task_rows.append(row)

    # ------------------------------------------------------------------
    # Write CSVs
    # ------------------------------------------------------------------
    with open(OUT / "per_model.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "model", "n", "fail", "p",
                    "p_easy", "p_medium", "p_hard"])
        for name, s in per_model.items():
            row = [name, s["model"], s["n"], s["n_fail"], f"{s['p']:.4f}"]
            for d in ("easy", "medium", "hard"):
                pd = s["p_by_difficulty"].get(d, 0.0)
                row.append(f"{pd:.4f}")
            w.writerow(row)

    with open(OUT / "pairwise.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "n",
                    "p_a", "p_b", "q",
                    "q_indep_if_uncorrelated", "q_over_min_p", "R_pair"])
        for (a, b), v in pairwise.items():
            w.writerow([a, b, v["n"],
                        f"{v['p_a']:.4f}", f"{v['p_b']:.4f}", f"{v['q']:.4f}",
                        f"{v['q_indep_if_uncorrelated']:.4f}",
                        f"{v['q_over_min_p']:.4f}", f"{v['R_pair']:.4f}"])

    with open(OUT / "per_task.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "difficulty"] + names)
        w.writeheader()
        for row in per_task_rows:
            w.writerow(row)

    # ------------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------------
    lines = []
    lines.append("=" * 70)
    lines.append("PILOT STUDY RESULTS — Blocksworld planning")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Per-model marginal failure rates")
    lines.append("-" * 70)
    lines.append(f"{'run':30s} {'model':25s} {'fail/n':>8s} {'p':>7s}")
    for name, s in per_model.items():
        lines.append(f"{name:30s} {s['model'][:24]:25s} "
                     f"{s['n_fail']:>3d}/{s['n']:<4d}  {s['p']:>7.4f}")

    lines.append("")
    lines.append("Per-model failure rate by difficulty")
    lines.append("-" * 70)
    lines.append(f"{'run':30s}  {'easy':>10s}  {'medium':>10s}  {'hard':>10s}")
    for name, s in per_model.items():
        e = s["p_by_difficulty"].get("easy", 0.0)
        m = s["p_by_difficulty"].get("medium", 0.0)
        h = s["p_by_difficulty"].get("hard", 0.0)
        lines.append(f"{name:30s}  {e:>10.4f}  {m:>10.4f}  {h:>10.4f}")

    lines.append("")
    lines.append("Pairwise joint failure")
    lines.append("-" * 70)
    lines.append(f"{'pair':50s} {'p_a':>6s} {'p_b':>6s} {'q':>6s} "
                 f"{'q/min':>6s} {'R_pair':>8s}")
    for (a, b), v in pairwise.items():
        pair = f"{a} | {b}"
        qm = v["q_over_min_p"]
        lines.append(f"{pair:50s} {v['p_a']:>6.3f} {v['p_b']:>6.3f} "
                     f"{v['q']:>6.3f} {qm:>6.3f} {v['R_pair']:>8.4f}")

    lines.append("")
    lines.append("Per-task pass/fail (P=pass, F=fail)")
    lines.append("-" * 70)
    header = f"{'task':8s} {'diff':8s}  " + "  ".join(f"{n[:12]:>12s}" for n in names)
    lines.append(header)
    for row in per_task_rows:
        cells = "  ".join(f"{row[n]:>12s}" for n in names)
        lines.append(f"{row['task']:8s} {row['difficulty']:8s}  {cells}")

    summary_txt = "\n".join(lines)
    (OUT / "summary.txt").write_text(summary_txt)
    print(summary_txt)

    # ------------------------------------------------------------------
    # JSON for LaTeX
    # ------------------------------------------------------------------
    blob = {
        "per_model": per_model,
        "pairwise": [v for v in pairwise.values()],
        "per_task": per_task_rows,
    }
    (OUT / "summary.json").write_text(json.dumps(blob, indent=2))


if __name__ == "__main__":
    main()
