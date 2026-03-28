#!/usr/bin/env python3
"""
inference.py — LLM-powered inference script for SQL Query Review OpenEnv.

Must be placed in the ROOT directory of the project (per submission rules).

Required environment variables:
    API_BASE_URL   The base URL of the running OpenEnv environment.
                   e.g. https://neo0110-openenv-sql-review.hf.space
    MODEL_NAME     The LLM model identifier.
                   e.g. gpt-4o-mini
    HF_TOKEN       Your Hugging Face / OpenAI API key.

Usage:
    export API_BASE_URL=https://neo0110-openenv-sql-review.hf.space
    export MODEL_NAME=gpt-4o-mini
    export HF_TOKEN=sk-...
    python inference.py

Results are written to baseline_results.json.
Runtime target: < 20 minutes on vcpu=2, memory=8GB.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# Config — reads the three required submission env vars
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL   = os.getenv("API_BASE_URL", "http://localhost:7860").rstrip("/")
MODEL      = os.getenv("MODEL_NAME", "gpt-4o-mini")
# HF_TOKEN is the required var name; fall back to OPENAI_API_KEY for local dev
API_KEY    = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")

SYSTEM_PROMPT = """\
You are a senior data engineer conducting a rigorous SQL code review.

You will be given:
- A SQL query to review
- The database schema (tables, columns, types, indexes)
- Task instructions

Your job:
1. Identify ALL correctness bugs, performance anti-patterns, and security vulnerabilities.
2. For each issue provide: category, severity, description, and suggested_fix.
3. After identifying issues, submit a final decision: approve | request_changes | reject.

Categories: correctness | performance | security | style
Severities: critical | high | medium | low

Rules:
- Only report real issues — do not invent issues that are not present.
- A critical severity means data loss, corruption, or security breach is possible.
- Reject if any critical issue exists; request_changes for high/medium; approve only if clean.
- Be concise but precise in descriptions.

Respond ONLY with valid JSON in this exact format:
{
  "issues": [
    {
      "category": "...",
      "severity": "...",
      "description": "...",
      "suggested_fix": "..."
    }
  ],
  "comment": "...",
  "decision": "approve|request_changes|reject"
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _env_post(path: str, **kwargs) -> dict:
    r = requests.post(f"{BASE_URL}{path}", **kwargs, timeout=30)
    r.raise_for_status()
    return r.json()

def _env_get(path: str, **kwargs) -> dict:
    r = requests.get(f"{BASE_URL}{path}", **kwargs, timeout=30)
    r.raise_for_status()
    return r.json()

def _build_user_message(obs: dict) -> str:
    schema_text = ""
    for tbl in obs.get("schema_context", []):
        cols = ", ".join(
            f"{c['name']} {c['type']}{'(nullable)' if c.get('nullable', True) else ' NOT NULL'}"
            for c in tbl["columns"]
        )
        indexes = ", ".join(tbl.get("has_index_on", []))
        constraints = "; ".join(tbl.get("constraints", []))
        row_hint = tbl.get("row_count_hint", 0)
        schema_text += (
            f"\nTable: {tbl['table_name']} (~{row_hint:,} rows)\n"
            f"  Columns: {cols}\n"
            f"  Indexes: {indexes or 'none'}\n"
        )
        if constraints:
            schema_text += f"  Constraints: {constraints}\n"

    return (
        f"## Task Instructions\n{obs['task_instructions']}\n\n"
        f"## SQL Query ({obs.get('query_dialect','PostgreSQL')})\n"
        f"```sql\n{obs['query']}\n```\n\n"
        f"## Schema\n{schema_text}"
    )

def _call_llm(client: OpenAI, user_msg: str, retry: int = 3) -> dict:
    for attempt in range(retry):
        try:
            response = client.chat.completions.create(
                model    = MODEL,
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature = 0.1,
                max_tokens  = 1200,
            )
            raw = response.choices[0].message.content or ""
            raw = raw.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = "\n".join(raw.split("\n")[:-1])
            return json.loads(raw.strip())
        except json.JSONDecodeError as e:
            print(f"  [warn] JSON parse error (attempt {attempt+1}): {e}")
            time.sleep(1)
        except Exception as e:
            print(f"  [warn] LLM call error (attempt {attempt+1}): {e}")
            time.sleep(2)
    return {"issues": [], "comment": "LLM failed to respond.", "decision": "approve"}


# ─────────────────────────────────────────────────────────────────────────────
# Run one task
# ─────────────────────────────────────────────────────────────────────────────

def run_task(task_id: str, client: OpenAI) -> dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print(f"{'='*60}")

    obs = _env_post("/reset", params={"task_id": task_id})
    session_id = obs["session_id"]
    print(f"Session: {session_id}  (max_steps={obs['max_steps']})")

    step_results = []
    done = obs.get("done", False)
    step = 0

    while not done and step < obs["max_steps"]:
        step += 1
        print(f"\n--- Step {step} ---")

        user_msg = _build_user_message(obs)
        llm_resp = _call_llm(client, user_msg)

        valid_cats = {"correctness", "performance", "security", "style"}
        valid_sevs = {"critical", "high", "medium", "low"}
        valid_decs = {"approve", "request_changes", "reject"}

        cleaned_issues = []
        for iss in llm_resp.get("issues", []):
            cat = iss.get("category", "correctness").lower()
            sev = iss.get("severity", "medium").lower()
            if cat not in valid_cats:
                cat = "correctness"
            if sev not in valid_sevs:
                sev = "medium"
            cleaned_issues.append({
                "category":     cat,
                "severity":     sev,
                "description":  iss.get("description", ""),
                "suggested_fix": iss.get("suggested_fix", ""),
            })

        decision = llm_resp.get("decision", "").lower()
        if decision not in valid_decs:
            decision = None

        action = {
            "session_id": session_id,
            "issues":     cleaned_issues,
            "comment":    llm_resp.get("comment", ""),
            "decision":   decision,
        }

        print(f"  Issues submitted: {len(cleaned_issues)}")
        print(f"  Decision: {decision}")

        result = _env_post("/step", json=action)
        obs    = result["observation"]
        reward = result["reward"]
        done   = result["done"]

        print(f"  Reward: {reward['value']:.4f}  (cumulative={reward['cumulative']:.4f})")
        step_results.append({
            "step":     step,
            "n_issues": len(cleaned_issues),
            "decision": decision,
            "reward":   reward["value"],
        })

    grader = _env_get("/grader", params={"session_id": session_id})
    print(f"\nFinal score: {grader['score']:.4f}")

    return {
        "task_id":      task_id,
        "session_id":   session_id,
        "score":        grader["score"],
        "breakdown":    grader.get("breakdown", {}),
        "step_results": step_results,
        "model":        MODEL,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run LLM inference against SQL Review OpenEnv")
    parser.add_argument("--task", default="all", help="Task ID or 'all'")
    parser.add_argument("--output", default="baseline_results.json")
    args = parser.parse_args()

    # Validate required env vars
    missing = []
    if not API_KEY:
        missing.append("HF_TOKEN (or OPENAI_API_KEY)")
    if not os.getenv("API_BASE_URL"):
        print(f"[warn] API_BASE_URL not set, defaulting to {BASE_URL}")
    if not os.getenv("MODEL_NAME"):
        print(f"[warn] MODEL_NAME not set, defaulting to {MODEL}")
    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

    # Verify environment is reachable
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=10).json()
        print(f"Environment: {BASE_URL}  status={health.get('status')}")
    except Exception as e:
        print(f"ERROR: Cannot reach environment at {BASE_URL}: {e}")
        sys.exit(1)

    client = OpenAI(api_key=API_KEY)

    # Determine tasks to run
    if args.task == "all":
        tasks_response = requests.get(f"{BASE_URL}/tasks", timeout=10).json()
        task_ids = [t["id"] for t in tasks_response["tasks"]]
    else:
        task_ids = [args.task]

    print(f"Running {len(task_ids)} task(s) with model={MODEL}\n")

    all_results = []
    for tid in task_ids:
        result = run_task(tid, client)
        all_results.append(result)
        time.sleep(0.5)

    avg = round(sum(r["score"] for r in all_results) / len(all_results), 4)

    summary = {
        "model":         MODEL,
        "environment":   BASE_URL,
        "average_score": avg,
        "results":       all_results,
    }

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Average score: {avg:.4f}")
    print(f"Results written to: {args.output}")
    print(f"{'='*60}")
    print(f"\n{'Task':<30} {'Score':>6}")
    print("-" * 38)
    for r in all_results:
        print(f"{r['task_id']:<30} {r['score']:>6.4f}")


if __name__ == "__main__":
    main()
