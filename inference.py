#!/usr/bin/env python3
"""
inference.py — OpenAI-powered agent against the SQL Query Review environment.

Usage:
    export OPENAI_API_KEY=sk-...
    export ENV_BASE_URL=http://localhost:7860   # or HF Space URL
    python scripts/inference.py

The agent is given the query + schema and prompted to act as a senior
data engineer performing a code review. It iterates up to max_steps,
submitting issues and a final decision.

Expected approximate scores (gpt-4o-mini):
  easy_correctness   → 0.70–0.85
  medium_performance → 0.55–0.75
  hard_security      → 0.45–0.65
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
MAX_STEPS = 6

client = OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────

def env_reset(task_id: str) -> Dict:
    resp = requests.post(f"{BASE_URL}/reset", params={"task_id": task_id})
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict) -> Dict:
    resp = requests.post(f"{BASE_URL}/step", json=action)
    resp.raise_for_status()
    return resp.json()


def env_grade(session_id: str) -> float:
    resp = requests.get(f"{BASE_URL}/grader", params={"session_id": session_id})
    resp.raise_for_status()
    return resp.json()["score"]


# ─────────────────────────────────────────────
# LLM agent
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior data engineer performing SQL code review.

You will be given a SQL query and schema context. Your job is to:
1. Identify ALL issues: correctness bugs, performance anti-patterns, security vulnerabilities.
2. Rate each issue by severity: critical | high | medium | low.
3. Categorize each issue: correctness | performance | security | style.
4. Suggest a fix where possible.
5. Submit a final decision: approve | request_changes | reject.

DECISION RULES:
- reject: any critical security or data-integrity issue
- request_changes: high/medium issues but no critical ones  
- approve: only if no real issues found

Respond ONLY with valid JSON in this exact format:
{
  "issues": [
    {
      "category": "security|correctness|performance|style",
      "severity": "critical|high|medium|low",
      "line_hint": null,
      "description": "Clear description of the issue",
      "suggested_fix": "How to fix it"
    }
  ],
  "comment": "Brief overall review summary",
  "decision": "approve|request_changes|reject"
}
"""


def build_user_prompt(obs: Dict) -> str:
    schema_str = ""
    for tbl in obs.get("schema_context", []):
        cols = ", ".join(f"{c['name']} {c['type']}" for c in tbl["columns"])
        idx = ", ".join(tbl.get("has_index_on", []))
        rows = tbl.get("row_count_hint", "unknown")
        schema_str += f"\nTable: {tbl['table_name']} (~{rows:,} rows)\n  Columns: {cols}\n  Indexes: {idx}\n"

    prior = ""
    for item in obs.get("review_thread", []):
        if item.get("type") == "issue":
            prior += f"  - [{item['severity']}] {item['category']}: {item['description']}\n"

    return f"""Task: {obs['task_instructions']}

=== SQL Query ({obs['query_dialect']}) ===
{obs['query']}

=== Schema Context ===
{schema_str}

=== Issues Already Identified ===
{prior if prior else '  (none yet)'}

Identify ALL remaining issues and provide your final decision."""


def run_agent(task_id: str) -> Dict:
    print(f"\n{'='*60}")
    print(f"Task: {task_id}")

    obs = env_reset(task_id)
    session_id = obs["session_id"]
    print(f"Session: {session_id}")

    step = 0
    done = False
    cumulative_reward = 0.0
    all_issues = []

    while not done and step < MAX_STEPS:
        step += 1
        prompt = build_user_prompt(obs)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            agent_action = json.loads(raw)
        except Exception as e:
            print(f"  LLM error at step {step}: {e}")
            break

        # Build action
        action = {
            "session_id": session_id,
            "issues": agent_action.get("issues", []),
            "comment": agent_action.get("comment"),
            "decision": agent_action.get("decision"),
        }

        result = env_step(action)
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        cumulative_reward = reward["cumulative"]
        all_issues.extend(agent_action.get("issues", []))

        print(f"  Step {step}: reward={reward['value']:.3f} cumulative={cumulative_reward:.3f} done={done}")
        if agent_action.get("decision"):
            print(f"  Decision: {agent_action['decision']}")

    # Grade
    score = env_grade(session_id)
    print(f"  Final score: {score:.4f}")
    print(f"  Issues found: {len(all_issues)}")

    return {
        "task_id": task_id,
        "score": score,
        "steps": step,
        "cumulative_reward": cumulative_reward,
        "issues_found": len(all_issues),
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    # Verify environment is reachable
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=10)
        resp.raise_for_status()
        print(f"Environment health: {resp.json()}")
    except Exception as e:
        print(f"Cannot reach environment at {BASE_URL}: {e}")
        sys.exit(1)

    tasks = ["easy_correctness", "medium_performance", "hard_security"]
    results = []

    for task_id in tasks:
        try:
            result = run_agent(task_id)
            results.append(result)
            time.sleep(1)
        except Exception as e:
            print(f"Error on {task_id}: {e}")
            results.append({"task_id": task_id, "score": 0.0, "error": str(e)})

    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    for r in results:
        score = r.get("score", 0.0)
        print(f"  {r['task_id']:30s}  score={score:.4f}")

    avg = sum(r.get("score", 0.0) for r in results) / len(results)
    print(f"\n  Average score: {avg:.4f}")
    print(f"{'='*60}\n")

    # Write results JSON for CI
    with open("baseline_results.json", "w") as f:
        json.dump({"results": results, "average": avg}, f, indent=2)
    print("Results written to baseline_results.json")


if __name__ == "__main__":
    main()
