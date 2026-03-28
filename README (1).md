---
title: SQL Query Review OpenEnv
emoji: 🗄️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - sql
  - code-review
app_port: 7860
---

# SQL Query Review — OpenEnv Environment v2.0

[![OpenEnv](https://img.shields.io/badge/spec-openenv--v1-blue)](https://openenv.dev)
[![HF Space](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces/Neo0110/openenv-sql-review)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tasks](https://img.shields.io/badge/tasks-6%20fixed%20%2B%20∞%20generated-brightgreen)]()

---

## Overview

**SQL Query Review** is an OpenEnv-compliant reinforcement learning environment that trains and evaluates AI agents to perform **SQL code review** — a genuine, high-stakes task performed daily by data engineers and database administrators.

The agent acts as a senior data engineer who receives a SQL query alongside its full database schema. Its job is to:

1. **Identify issues** across correctness, performance, security, and style categories
2. **Rate severity** of each issue: `critical | high | medium | low`
3. **Submit a verdict**: `approve | request_changes | reject`

This simulates the exact workflow of a real data engineering pull-request review. Training agents on this environment develops skills directly transferable to automated code review tooling, developer copilots, and data platform quality gates.

---

## Why This Domain?

SQL code review is a **high-stakes, real-world problem**:

- **Correctness bugs** (wrong JOIN type, aggregation logic errors) silently corrupt analytics reports serving business decisions
- **Performance anti-patterns** (correlated subqueries, non-sargable predicates) take production systems offline
- **Security vulnerabilities** (SQL injection, credential leaks) lead to data breaches

No publicly available RL environment exists for this problem — yet the demand for automated SQL review tools is enormous (Postgres alone powers millions of production databases worldwide).

---

## What's New in v2.0

| Feature | v1 | v2 |
|---|---|---|
| Fixed tasks | 3 | **6** (easy×2, medium×2, hard×2) |
| Procedural task generation | ❌ | ✅ **unlimited novel tasks** |
| Issue matching | keyword regex | **semantic cosine similarity** — rewards paraphrases |
| Grader | basic | **weighted detection × FP multiplier × decision** |
| Schema detail | minimal | **nullable flags, constraints, row counts** |
| Tests | ~10 | **30+ tests** across all spec areas |
| Hard task calibration | fixed | **hard reliably scores lower than easy for agents** |

---

## Environment Architecture

```
POST /reset?task_id=...   →  Observation (query + schema + instructions)
POST /step                →  action → (Observation, Reward, done, info)
GET  /state               →  full episode state
GET  /grader              →  final 0.0–1.0 score
GET  /tasks               →  task list + action schema + procedural task info
POST /baseline            →  run heuristic baseline on all 6 tasks (no API key)
GET  /health              →  liveness probe
GET  /                    →  environment metadata
```

Interactive docs: `<your-space-url>/docs`

---

## Action & Observation Spaces

### Observation

| Field | Type | Description |
|---|---|---|
| `session_id` | string | Unique episode identifier |
| `task_id` | string | Which task is active |
| `step_number` | int | Current step (0-indexed) |
| `max_steps` | int | Steps until forced termination |
| `query` | string | The SQL query under review |
| `query_dialect` | string | SQL dialect (PostgreSQL) |
| `schema_context` | list[SchemaInfo] | Tables, columns (with nullable flags), row counts, indexes, constraints |
| `review_thread` | list[dict] | Prior agent comments/issues in this episode |
| `task_instructions` | string | What the agent should focus on |
| `done` | bool | Whether episode has ended |

**SchemaInfo** contains: `table_name`, `columns` (name + type + nullable), `row_count_hint`, `has_index_on`, `constraints`.

### Action

| Field | Type | Description |
|---|---|---|
| `session_id` | string | Must match the current episode |
| `issues` | list[ReviewIssue] | Identified issues this step |
| `comment` | string (optional) | Overall review comment |
| `decision` | enum (optional) | `approve / request_changes / reject` — submitting this ends the episode |

**ReviewIssue** contains: `category` (correctness/performance/security/style), `severity` (critical/high/medium/low), `line_hint` (optional int), `description`, `suggested_fix` (optional).

### Reward

Dense reward provided at every step:

| Component | Max Value | Trigger |
|---|---|---|
| Issue detection | `+0.50` | Newly matched ground-truth issue (weighted by category+severity) |
| Severity accuracy | `+0.10` | Severity within 1 level of ground truth |
| False positive penalty | `−0.12` per FP | Fabricated / irrelevant issue |
| Decision reward | `+0.20` | Correct final verdict |
| Partial decision | `+0.05` | Near-correct (request_changes ↔ reject) |
| Efficiency bonus | `+0.01×steps_saved` | Finishing before max_steps |

Reward range: `[−1.0, +1.0]` per step. Final grader score: `[0.0, 1.0]`.

#### Issue Matching — Semantic Similarity

v2 uses **TF-IDF cosine similarity** to match agent-submitted issues against ground truth. This means agents are rewarded for correct descriptions in their own words — not just for hitting specific keywords. A similarity threshold of 0.22 separates true positives from noise, tuned to reward natural paraphrases while blocking irrelevant submissions.

---

## Tasks

### Task 1 — Easy: Wrong JOIN Type Bug
**ID:** `easy_correctness` | **Max steps:** 5

A monthly report query uses `INNER JOIN` between `orders` and `customers`. The schema shows `orders.customer_id` is **nullable** (guest checkouts have no customer record). The `INNER JOIN` silently drops all guest orders from the report.

- **Ground truth issues:** 1 (correctness/high)
- **Correct decision:** `request_changes`
- **What an agent must catch:** Wrong JOIN type on nullable FK

---

### Task 2 — Easy: Implicit Type Coercion
**ID:** `easy_type_coercion` | **Max steps:** 5

A user lookup compares a `VARCHAR(20)` column (`account_code`) to an integer literal. The implicit coercion defeats the B-tree index on a 5M-row table and can silently miss leading-zero codes like `'010042'`.

- **Ground truth issues:** 2 (performance/high + correctness/medium)
- **Correct decision:** `request_changes`
- **What an agent must catch:** Non-sargable predicate from type mismatch

---

### Task 3 — Medium: Three Performance Anti-Patterns
**ID:** `medium_performance` | **Max steps:** 7

A dashboard query runs every minute against 50M-row `users` and 200M-row `orders` tables. It has three distinct performance problems:

1. **Correlated subquery** — N+1 pattern, executes once per outer row
2. **`LOWER(email)` on indexed column** — non-sargable, causes full table scan
3. **`SELECT u.*`** — pulls all columns unnecessarily on a 50M-row table

- **Ground truth issues:** 3 (performance/high, performance/high, performance/medium)
- **Correct decision:** `request_changes`
- **What an agent must catch:** All three anti-patterns with correct severity

---

### Task 4 — Medium: Aggregation Logic + Missing Index
**ID:** `medium_aggregation` | **Max steps:** 7

A weekly revenue report has four compounding bugs: a no-op `HAVING` clause, a miscalculated `AVG` metric, a missing index on the `WHERE` column, and an off-by-one date boundary with `BETWEEN`.

- **Ground truth issues:** 4 (correctness/high, correctness/medium, performance/high, correctness/low)
- **Correct decision:** `request_changes`
- **What an agent must catch:** Logic errors in aggregation + query plan issues

---

### Task 5 — Hard: Security + Logic + Performance (Stored Function)
**ID:** `hard_security` | **Max steps:** 10

A public-facing product search function builds SQL dynamically. Four bugs span three categories:

1. **SQL injection** (critical) — `search_term` interpolated directly via `format('%s', ...)`
2. **Credential leak** (critical) — `password_hash` selected and returned to API callers
3. **Wrong aggregation filter** (high) — `HAVING COUNT > 0` + `LEFT JOIN` hides unreviewed products
4. **Unindexed sort DoS** (medium) — user-controlled `sort_by` allows arbitrary filesort

- **Ground truth issues:** 4 (security/critical × 2, correctness/high, performance/medium)
- **Correct decision:** `reject`
- **What an agent must catch:** Both critical security issues required for high score

---

### Task 6 — Hard: Destructive Schema Migration
**ID:** `hard_migration` | **Max steps:** 10

A production migration script splits `full_name` into `first_name`/`last_name` on a 12M-row table. Multiple risks are present:

1. **Data loss for single-name users** (critical) — `SPLIT_PART` returns `''` for "Madonna", then `NOT NULL` constraint explodes
2. **Irreversible `DROP COLUMN`** (high) — no backup strategy before destructive operation
3. **Table-locking `SET NOT NULL`** (high) — ACCESS EXCLUSIVE lock blocks all reads for minutes
4. **Missing FK constraint on audit log** (medium) — orphaned rows possible

- **Ground truth issues:** 4 (correctness/critical, correctness/high, performance/high, correctness/medium)
- **Correct decision:** `reject`
- **What an agent must catch:** Data-loss risk and locking implications in production migration

---

### Procedural Task Generation

Pass `task_id=generated_<difficulty>_<seed>` to `/reset` for unlimited novel tasks:

```bash
# Generate a novel hard task
curl -X POST "http://localhost:7860/reset?task_id=generated_hard_42"

# Same seed = same task every time (deterministic)
curl -X POST "http://localhost:7860/reset?task_id=generated_medium_7"
```

Procedurally generated tasks compose bug templates with randomised table/column names and schemas, ensuring agents cannot simply memorise the fixed task set.

---

## Reward Design Details

The reward function is **dense** — meaningful signal at every step.

**Key design choices:**

- **Semantic matching via cosine similarity** — agents rewarded for correct paraphrases, not just keyword matches
- **Weighted detection** — security bugs (weight 1.4×) and correctness bugs (1.1×) valued more than style (0.6×)
- **Severity weight scaling** — critical issues (1.5×) worth more than low issues (0.3×)
- **False-positive penalty** (−0.12 per FP) — discourages hallucinating issues to pad scores
- **Aggressive grader FP multiplier** — `max(0.4, 1 − fp_rate × 1.5)` severely punishes hallucination at grader time
- **Partial decision credit** — `request_changes` vs `reject` gets 0.05 partial, preventing binary cliff

### Final Grader Formula

```
detection_score  = Σ(weight of detected GT issues) / Σ(weight of all GT issues)
fp_rate          = false_positives / total_submitted
fp_multiplier    = max(0.4,  1 − fp_rate × 1.5)
decision_score   = 1.0 | 0.3 | 0.0

final_score = detection_score × fp_multiplier × 0.80
            + decision_score  × 0.20
```

---

## Baseline Scores

### Heuristic Baseline (no LLM, fully deterministic)

| Task | Difficulty | Score |
|---|---|---|
| `easy_correctness` | Easy | ~0.80 |
| `easy_type_coercion` | Easy | ~0.72 |
| `medium_performance` | Medium | ~0.55 |
| `medium_aggregation` | Medium | ~0.48 |
| `hard_security` | Hard | ~0.65 |
| `hard_migration` | Hard | ~0.58 |
| **Average** | | **~0.63** |

*Note: hard tasks score lower than easy on average — difficulty ordering is correct and meaningful.*

### LLM Baseline (gpt-4o-mini, approximate)

| Task | Difficulty | Expected Score |
|---|---|---|
| `easy_correctness` | Easy | ~0.82–0.90 |
| `easy_type_coercion` | Easy | ~0.70–0.80 |
| `medium_performance` | Medium | ~0.60–0.75 |
| `medium_aggregation` | Medium | ~0.50–0.65 |
| `hard_security` | Hard | ~0.50–0.65 |
| `hard_migration` | Hard | ~0.45–0.60 |
| **Average** | | **~0.60–0.72** |

---

## Setup & Usage

### Option 1 — Docker (recommended)

```bash
git clone https://huggingface.co/spaces/Neo0110/openenv-sql-review
cd openenv-sql-review

docker build -t sql-review-env .
docker run -p 7860:7860 sql-review-env
```

Environment available at `http://localhost:7860`. Interactive docs at `http://localhost:7860/docs`.

### Option 2 — Local Python

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

### Running the Heuristic Baseline (no API key needed)

```bash
# Via HTTP endpoint (deterministic, fully reproducible):
curl -X POST http://localhost:7860/baseline | python -m json.tool
```

### Running the LLM Inference Script

```bash
export OPENAI_API_KEY=sk-...
export ENV_BASE_URL=http://localhost:7860   # or your HF Space URL
python scripts/inference.py                # runs all 6 tasks
python scripts/inference.py --task hard_security   # single task
```

Results written to `baseline_results.json`.

### Running Tests

```bash
pip install pytest httpx
pytest tests/ -v
# Expected: 30+ tests, all passing
```

---

## Example Agent Interaction

```python
import requests

BASE = "http://localhost:7860"

# 1. Start an episode
obs = requests.post(f"{BASE}/reset", params={"task_id": "hard_security"}).json()
session_id = obs["session_id"]
print("Query:", obs["query"][:120], "...")

# 2. Submit a multi-issue review
action = {
    "session_id": session_id,
    "issues": [
        {
            "category": "security",
            "severity": "critical",
            "description": "search_term is injected directly into format('%s', search_term) — SQL injection. "
                           "Attacker can execute arbitrary SQL.",
            "suggested_fix": "Use EXECUTE ... USING $1 with parameterised queries."
        },
        {
            "category": "security",
            "severity": "critical",
            "description": "password_hash is selected and returned — leaks all seller password hashes to API callers.",
            "suggested_fix": "Remove password_hash from SELECT and RETURNS TABLE signature."
        },
    ],
    "comment": "Two critical security bugs. This function must not be deployed.",
    "decision": "reject"
}
result = requests.post(f"{BASE}/step", json=action).json()
print("Reward:", result["reward"]["value"])
print("Done:", result["done"])

# 3. Get final grade with breakdown
grade = requests.get(f"{BASE}/grader", params={"session_id": session_id}).json()
print("Score:", grade["score"])
print("Breakdown:", grade["breakdown"])

# 4. Try a procedurally generated task
obs2 = requests.post(f"{BASE}/reset", params={"task_id": "generated_hard_42"}).json()
print("Novel task:", obs2["task_id"], "| Query:", obs2["query"][:80], "...")
```

---

## Project Structure

```
openenv-sql-review/
├── app/
│   ├── __init__.py
│   ├── main.py        # FastAPI app — all HTTP endpoints
│   ├── models.py      # Pydantic: Observation, Action, Reward, GraderResult
│   ├── tasks.py       # 6 fixed tasks + procedural generator
│   ├── session.py     # Episode state, semantic reward shaping, grader
│   └── baseline.py    # Heuristic baseline (10 regex rules, no LLM)
├── scripts/
│   └── inference.py   # LLM-powered inference (OpenAI client)
├── tests/
│   └── test_environment.py   # 30+ tests covering full spec + edge cases
├── openenv.yaml       # OpenEnv spec metadata (v2)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Validation Checklist

```bash
curl http://localhost:7860/health                     # → {"status":"ok"}
curl http://localhost:7860/                           # → root metadata
curl http://localhost:7860/tasks                      # → 6 tasks + action schema
curl -X POST http://localhost:7860/reset              # → Observation
curl -X POST http://localhost:7860/baseline           # → 6 scores, avg ~0.63
pytest tests/ -v                                      # → 30+ tests passing
docker build -t sql-review-env . && docker run -p 7860:7860 sql-review-env
```

---

## License

MIT