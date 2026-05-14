"""
Evaluate OpenAI models on the Blocksworld pilot tasks.

Loads OPENAI_API_KEY from pilot/.env (or process env), sends each task to the
selected OpenAI model(s) with temperature=0, and saves each response to
answers/openai_<model>/bw-XX.txt.

Usage:
    .venv/bin/python eval_openai.py
    .venv/bin/python eval_openai.py --models gpt-4o-mini gpt-4o
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# --- load .env --------------------------------------------------------------
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
    import openai
except ImportError:
    print("ERROR: openai SDK not installed. Run:")
    print("  .venv/bin/pip install openai")
    sys.exit(1)


# --- configuration ----------------------------------------------------------

# Default models to test. Edit as needed.
DEFAULT_MODELS = ["gpt-4o-mini", "gpt-4o"]

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


# --- core -------------------------------------------------------------------

def run_task(client, model_id: str, task_text: str) -> tuple[str, dict]:
    t0 = time.time()
    # Retry on rate-limit with exponential backoff
    delay = 25.0
    for attempt in range(6):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task_text},
                ],
            )
            break
        except openai.RateLimitError as e:
            if attempt == 5:
                raise
            print(f"    rate-limited (attempt {attempt + 1}); sleeping {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 1.5, 90.0)
    elapsed = time.time() - t0
    text = resp.choices[0].message.content or ""
    meta = {
        "model": model_id,
        "finish_reason": resp.choices[0].finish_reason,
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "elapsed_sec": round(elapsed, 2),
    }
    return text, meta


def evaluate_one(client, model_id: str, task_id: str, out_dir: Path,
                 skip_existing: bool = True) -> dict:
    out_txt = out_dir / f"{task_id}.txt"
    out_meta = out_dir / f"{task_id}.meta.json"
    if skip_existing and out_txt.exists() and out_meta.exists():
        meta = json.loads(out_meta.read_text())
        return {"task_id": task_id, "ok": True, "skipped": True, **meta}
    task_path = TASKS_DIR / f"{task_id}.txt"
    if not task_path.exists():
        return {"task_id": task_id, "ok": False, "error": "task file not found"}
    task_text = task_path.read_text()
    try:
        text, meta = run_task(client, model_id, task_text)
    except Exception as e:
        return {"task_id": task_id, "ok": False, "error": f"{type(e).__name__}: {e}"}
    out_txt.write_text(text)
    out_meta.write_text(json.dumps(meta, indent=2))
    return {"task_id": task_id, "ok": True, **meta}


def safe_dirname(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                   help="OpenAI model IDs to evaluate")
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--concurrency", type=int, default=1,
                   help="parallel requests per model (default 1 for free tier)")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set (check pilot/.env)")
        sys.exit(1)

    client = openai.OpenAI()

    if args.tasks:
        task_ids = args.tasks
    else:
        task_ids = sorted(t.stem for t in TASKS_DIR.glob("bw-*.txt"))
    print(f"Tasks: {len(task_ids)}")

    overall = {}
    for model_id in args.models:
        print(f"\n=== Evaluating {model_id} ===")
        out_dir = ANSWERS_DIR / f"openai_{safe_dirname(model_id)}"
        out_dir.mkdir(exist_ok=True)
        results = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(evaluate_one, client, model_id, tid, out_dir): tid
                    for tid in task_ids}
            for f in as_completed(futs):
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
            "model": model_id,
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
        overall[model_id] = summary
        print(f"  Completed {summary['n_ok']}/{summary['n_tasks']}")

    print("\n=== Overall ===")
    for mid, s in overall.items():
        print(f"  {mid}  {s['n_ok']}/{s['n_tasks']}")


if __name__ == "__main__":
    main()
