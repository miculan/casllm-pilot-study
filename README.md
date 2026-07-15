# CASLLM Pilot Study — Blocksworld Planning with LLMs

Empirical companion to the paper *Reliable Decision Loops in Socio-Technical
Collective Adaptive Systems* (M. Miculan). This repository contains the code,
tasks, raw model responses, and aggregated results of a pilot study that
measures **marginal and joint failure rates** of five state-of-the-art LLMs on
a small Blocksworld planning benchmark.

The goal is not to benchmark planning ability per se, but to probe the
**correlation structure** of LLM failures: how often do two models fail on the
*same* task? This joint failure probability $q$ is the quantity that drives
the reliability of guarded decision loops in the accompanying paper.

## Models evaluated

| Family | Identifier used                        |
|--------|----------------------------------------|
| Claude | `claude-haiku-4-5`, `claude-sonnet-4-6`, `claude-opus-4-7` |
| OpenAI | `gpt-4o-mini`, `gpt-4o`                |

## Benchmark

30 Blocksworld tasks (`tasks/bw-01` … `bw-30`), evenly split:

- **easy** (10): 3 blocks
- **medium** (10): 4–5 blocks
- **hard** (10): 6–8 blocks

Each task is rendered both as a plain-text prompt (`bw-XX.txt`, shown to the
LLM) and as a machine-readable spec (`bw-XX.json`, consumed by the validator).
The reference optimal plan length is recorded but never shown to the model.

## What is measured

The validator (`validator.py`) executes the LLM's plan against a faithful
Blocksworld simulator and checks that the goal state is reached. From the
per-task pass/fail outcomes, `analyze.py` computes:

- **marginal failure rate** $p_i$ for each model;
- **joint failure rate** $q(i,j)$ for every pair, i.e. the fraction of tasks
  on which *both* models fail;
- the ratio $q /\!\min(p_i, p_j)$, which quantifies how concentrated the
  overlap is on the weaker component;
- the pair-reliability $R_\text{pair} = 1 - q$, i.e. the probability that at
  least one model in the pair produces a correct plan;
- per-difficulty breakdowns and a per-task pass/fail table.

## Layout

```
casllm-pilot-study/
├── generate_tasks.py         # writes the 30 tasks to tasks/
├── eval_claude.py            # queries Claude models, writes answers/claude_*/
├── eval_openai.py            # queries OpenAI models, writes answers/openai_*/
├── validator.py              # Blocksworld plan validator (single + --batch)
├── test_validator.py         # smoke tests for the validator
├── analyze.py                # aggregates pairwise q, R_pair, etc.
├── tasks/                    # bw-XX.txt + bw-XX.json (30 tasks)
├── answers/<run>/            # one folder per model; bw-XX.txt responses + summary.json
└── results/                  # per_model.csv, pairwise.csv, per_task.csv, summary.{txt,json}
```

## Reproducing the experiment

### 1. Setup

```bash
git clone https://github.com/miculan/casllm-pilot-study.git
cd casllm-pilot-study

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
$EDITOR .env   # set ANTHROPIC_API_KEY and OPENAI_API_KEY
```

### 2. (Re)generate the tasks

The 30 tasks are already checked in under `tasks/`. If you want to regenerate
them (or modify the benchmark in `generate_tasks.py`):

```bash
.venv/bin/python generate_tasks.py
```

### 3. Sanity-check the validator

```bash
.venv/bin/python test_validator.py
```

### 4. Query the models

Each call is sent with `temperature=0` (Claude Opus 4.7 ignores the
temperature parameter, but is otherwise treated identically). Responses are
saved verbatim under `answers/<run>/bw-XX.txt`, with per-call metadata in
`bw-XX.meta.json`. Re-runs of `eval_openai.py` skip tasks whose response file
already exists; `eval_claude.py` always overwrites.

```bash
# All three Claude families
.venv/bin/python eval_claude.py

# Both GPT-4o variants (set --concurrency 1 if you hit rate limits)
.venv/bin/python eval_openai.py --models gpt-4o-mini gpt-4o
```

Subsets are supported:

```bash
.venv/bin/python eval_claude.py --models sonnet --tasks bw-01 bw-12
```

### 5. Validate each run

The validator parses the model output (looking for a `PLAN:` header, falling
back to any list of well-formed actions), simulates the plan, and checks goal
satisfaction. Per-task results are written to `summary.json` inside each
answer folder.

```bash
for d in answers/*/; do
  .venv/bin/python validator.py --batch "$d"
done
```

### 6. Aggregate

```bash
.venv/bin/python analyze.py
```

This writes `results/per_model.csv`, `results/pairwise.csv`,
`results/per_task.csv`, `results/summary.txt`, and `results/summary.json` (the
last one is the source used to populate the tables in the paper).

## Headline numbers from the checked-in run

Marginal failure rates over all 30 tasks:

| Model               | fail / 30 | $p$    |
|---------------------|-----------|--------|
| `claude-opus-4-7`   | 0 / 30    | 0.000  |
| `claude-sonnet-4-6` | 4 / 30    | 0.133  |
| `claude-haiku-4-5`  | 10 / 30   | 0.333  |
| `gpt-4o`            | 10 / 30   | 0.333  |
| `gpt-4o-mini`       | 23 / 30   | 0.767  |

A subset of the pairwise joint failure rates $q$ and the resulting
pair-reliability $R_\text{pair} = 1 - q$:

| Pair                                  | $p_a$ | $p_b$ | $q$   | $R_\text{pair}$ |
|---------------------------------------|-------|-------|-------|-----------------|
| `claude-haiku` ∥ `claude-sonnet`      | 0.333 | 0.133 | 0.067 | 0.933 |
| `claude-haiku` ∥ `gpt-4o`             | 0.333 | 0.333 | 0.167 | 0.833 |
| `claude-sonnet` ∥ `gpt-4o`            | 0.133 | 0.333 | 0.000 | 1.000 |
| `gpt-4o` ∥ `gpt-4o-mini` (same vendor)| 0.333 | 0.767 | 0.300 | 0.700 |

The full pairwise table is in
[`results/pairwise.csv`](results/pairwise.csv) and the per-task pass/fail
matrix in [`results/per_task.csv`](results/per_task.csv).

## Methodological notes

- **Determinism.** All calls use `temperature=0`; results are not bit-exact
  reproducible across API versions, but failure rates are stable across runs
  on the order of $\pm 1$ task in our experience.
- **Plan extraction.** The validator is lenient: it accepts numbered or
  bulleted plans, with or without a `PLAN:` header, and tolerates markdown
  backticks around actions. Outputs that contain *no* recognisable
  action-line are counted as failures.
- **What counts as success.** The plan must execute without violating any
  Blocksworld precondition and must reach the goal state with the arm empty.
  Sub-optimal plans (longer than the reference) still count as successes.
- **Sample size.** $n=30$ is a pilot. Confidence intervals on the joint
  failure rates are wide; the study is intended to demonstrate the
  *direction* of the effect predicted by the framework, not to provide tight
  estimates.

## Citation

If you use this benchmark or the analysis scripts, please cite the
accompanying paper:

> M. Miculan. *Reliable Decision Loops in Socio-Technical Collective Adaptive
> Systems*. In: Rigorous Engineering of Collective Adaptive Systems (REoCAS),
> ISoLA 2026. Lecture Notes in Computer Science, Springer. To appear.

```bibtex
@inproceedings{Miculan2026DecisionLoops,
  author    = {Miculan, Marino},
  title     = {Reliable Decision Loops in Socio-Technical Collective Adaptive Systems},
  booktitle = {Leveraging Applications of Formal Methods, Verification and
               Validation (ISoLA 2026), Rigorous Engineering of Collective
               Adaptive Systems track},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026},
  note      = {To appear}
}
```

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). The
recorded model responses under `answers/` are excerpts of Anthropic and
OpenAI API outputs, redistributed here under their respective terms for the
sole purpose of scientific reproducibility.
