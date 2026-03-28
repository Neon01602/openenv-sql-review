#!/usr/bin/env python3
"""
LLM-powered inference script for SQL Query Review OpenEnv.

Usage:
    export OPENAI_API_KEY=sk-...
    export ENV_BASE_URL=https://neo0110-openenv-sql-review.hf.space   # or http://localhost:7860
    python scripts/inference.py [--task all|easy_correctness|...]

Results are written to baseline_results.json.
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
# Config
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL    = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
OPENAI_KEY  = os.getenv("OPENAI_API_KEY", "")
MODEL       = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

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
            # Strip markdown fences if present
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

    # Reset
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

        # Clamp to valid values
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
            "step":    step,
            "n_issues": len(cleaned_issues),
            "decision": decision,
            "reward":  reward["value"],
        })

    # Grade
    grader = _env_get("/grader", params={"session_id": session_id})
    print(f"\nFinal score: {grader['score']:.4f}")
    if "breakdown" in grader:
        bd = grader["breakdown"]
        print(f"  detection={bd.get('detection_score',0):.3f}  "
              f"fp_mult={bd.get('fp_multiplier',0):.3f}  "
              f"decision={bd.get('decision_score',0):.3f}  "
              f"gt={bd.get('gt_detected',0)}/{bd.get('gt_total',0)}")

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

    if not OPENAI_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_KEY)

    # Determine which tasks to run
    if args.task == "all":
        tasks_response = requests.get(f"{BASE_URL}/tasks", timeout=10).json()
        task_ids = [t["id"] for t in tasks_response["tasks"]]
    else:
        task_ids = [args.task]

    print(f"Running inference on {len(task_ids)} task(s) using model={MODEL}")
    print(f"Environment: {BASE_URL}\n")

    all_results = []
    for tid in task_ids:
        result = run_task(tid, client)
        all_results.append(result)
        time.sleep(0.5)  # polite delay

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

    # Print table
    print(f"\n{'Task':<30} {'Difficulty':<12} {'Score':>6}")
    print("-" * 52)
    for r in all_results:
        tid = r["task_id"]
        diff = r["breakdown"].get("difficulty", "—")
        print(f"{tid:<30} {diff:<12} {r['score']:>6.4f}")


if __name__ == "__main__":
    main()