"""
Evaluate Claude models on the Blocksworld pilot tasks.

Loads ANTHROPIC_API_KEY from pilot/.env (or process env), sends each task to
the selected Claude model(s) with temperature=0, and saves each response to
answers/claude_<model>/bw-XX.txt.

Usage:
    .venv/bin/python eval_claude.py                  # default: all 3 models
    .venv/bin/python eval_claude.py --models sonnet  # only one model
    .venv/bin/python eval_claude.py --tasks bw-01 bw-05  # subset of tasks
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# --- load .env (no external dep) -------------------------------------------
ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
if ENV_FILE.exists():
    for raw in ENV_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic SDK not installed. Run:")
    print("  .venv/bin/pip install anthropic")
    sys.exit(1)


# --- configuration ---------------------------------------------------------

MODELS = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-7",
}

# We will probe the API for the exact latest model IDs; the constants above
# are defaults. If a constant is rejected, the script tries some fallbacks.
MODEL_FALLBACKS = {
    "haiku":  ["claude-haiku-4-5", "claude-haiku-4-5-20251001"],
    "sonnet": ["claude-sonnet-4-6", "claude-sonnet-4-5", "claude-sonnet-4-5-20250929"],
    "opus":   ["claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5"],
}

SYSTEM_PROMPT = (
    "You are a planning agent for the Blocksworld domain. "
    "When given a task description, output a single plan as a numbered list of "
    "primitive actions, using exactly the format requested in the task. "
    "Do not output explanations, reasoning, or any text other than the plan."
)

TEMPERATURE = 0.0
MAX_TOKENS = 1024

TASKS_DIR = ROOT / "tasks"
ANSWERS_DIR = ROOT / "answers"
ANSWERS_DIR.mkdir(exist_ok=True)


# --- core ------------------------------------------------------------------

def find_working_model(client, family: str) -> str:
    """Probe the API to find a working model ID for the given family."""
    candidates = [MODELS.get(family, family)] + MODEL_FALLBACKS.get(family, [])
    tried = set()
    for cand in candidates:
        if cand in tried or not cand:
            continue
        tried.add(cand)
        try:
            client.messages.create(
                model=cand,
                max_tokens=4,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return cand
        except anthropic.NotFoundError:
            continue
        except anthropic.BadRequestError as e:
            if "model" in str(e).lower():
                continue
            raise
    raise RuntimeError(f"No working {family} model found. Tried: {tried}")


def run_task(client, model_id: str, task_text: str) -> tuple[str, dict]:
    """Send one task to Claude. Returns (response_text, metadata)."""
    t0 = time.time()
    kwargs = dict(
        model=model_id,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": task_text}],
    )
    # Opus 4.7+ deprecated temperature; send it only for older models.
    if "opus-4-7" not in model_id and "opus-4.7" not in model_id:
        kwargs["temperature"] = TEMPERATURE
    try:
        resp = client.messages.create(**kwargs)
    except anthropic.BadRequestError as e:
        if "temperature" in str(e).lower() and "temperature" in kwargs:
            kwargs.pop("temperature")
            resp = client.messages.create(**kwargs)
        else:
            raise
    elapsed = time.time() - t0

    # Extract text content (block 0 is typically text)
    text = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            text += block.text
    metadata = {
        "model": model_id,
        "stop_reason": resp.stop_reason,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "elapsed_sec": round(elapsed, 2),
    }
    return text, metadata


def evaluate_one(client, model_family: str, model_id: str, task_id: str,
                 out_dir: Path) -> dict:
    """Evaluate one task and save the response. Returns a result dict."""
    task_path = TASKS_DIR / f"{task_id}.txt"
    if not task_path.exists():
        return {"task_id": task_id, "ok": False, "error": "task file not found"}
    task_text = task_path.read_text()
    try:
        text, meta = run_task(client, model_id, task_text)
    except Exception as e:
        return {"task_id": task_id, "ok": False, "error": f"{type(e).__name__}: {e}"}
    (out_dir / f"{task_id}.txt").write_text(text)
    (out_dir / f"{task_id}.meta.json").write_text(json.dumps(meta, indent=2))
    return {"task_id": task_id, "ok": True, **meta}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                   choices=list(MODELS.keys()),
                   help="Claude model families to evaluate")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Task IDs (default: all bw-* in tasks/)")
    p.add_argument("--concurrency", type=int, default=5,
                   help="Number of parallel requests per model")
    args = p.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set (check pilot/.env)")
        sys.exit(1)

    client = anthropic.Anthropic()

    # Resolve task list
    if args.tasks:
        task_ids = args.tasks
    else:
        task_ids = sorted(t.stem for t in TASKS_DIR.glob("bw-*.txt"))
    print(f"Tasks: {len(task_ids)}")

    # Resolve model IDs
    print("\nResolving model IDs:")
    model_ids = {}
    for fam in args.models:
        try:
            mid = find_working_model(client, fam)
            model_ids[fam] = mid
            print(f"  {fam:8s} -> {mid}")
        except RuntimeError as e:
            print(f"  {fam:8s} -> SKIPPED ({e})")

    # Run each model
    overall = {}
    for fam, mid in model_ids.items():
        print(f"\n=== Evaluating Claude {fam} ({mid}) ===")
        out_dir = ANSWERS_DIR / f"claude_{fam}"
        out_dir.mkdir(exist_ok=True)

        results = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(evaluate_one, client, fam, mid, tid, out_dir): tid
                       for tid in task_ids}
            for f in as_completed(futures):
                r = f.result()
                results.append(r)
                tid = r["task_id"]
                if r["ok"]:
                    print(f"  {tid}  {r['elapsed_sec']:5.2f}s  "
                          f"in={r['input_tokens']} out={r['output_tokens']}")
                else:
                    print(f"  {tid}  ERROR: {r['error']}")

        results.sort(key=lambda x: x["task_id"])
        summary = {
            "family": fam,
            "model": mid,
            "system_prompt": SYSTEM_PROMPT,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "n_tasks": len(task_ids),
            "n_ok": sum(1 for r in results if r["ok"]),
            "total_input_tokens": sum(r.get("input_tokens", 0) for r in results),
            "total_output_tokens": sum(r.get("output_tokens", 0) for r in results),
            "results": results,
        }
        (out_dir / "run_metadata.json").write_text(json.dumps(summary, indent=2))
        overall[fam] = summary
        print(f"  Completed {summary['n_ok']}/{summary['n_tasks']} "
              f"({summary['total_input_tokens']} in / "
              f"{summary['total_output_tokens']} out tokens)")

    # Top-level summary
    print("\n=== Overall summary ===")
    for fam, s in overall.items():
        print(f"  {fam:8s}  {s['model']}  {s['n_ok']}/{s['n_tasks']} completed")
    print(f"\nResponses saved under {ANSWERS_DIR}/")
    print("Next: run validator in batch mode on each answer directory.")


if __name__ == "__main__":
    main()
