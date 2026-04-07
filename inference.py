#!/usr/bin/env python3
"""
inference.py — OpenAI-powered agent against the SQL Query Review environment.

Required environment variables:
    API_BASE_URL      Base URL used for BOTH the OpenEnv environment and LLM proxy.
    MODEL_NAME        The LLM model identifier.  e.g. gpt-4o-mini
    API_KEY           API key injected by the validator proxy.

Usage:
    export API_BASE_URL=https://neo0110-openenv-sql-review.hf.space
    export MODEL_NAME=gpt-4o-mini
    export API_KEY=sk-...
    python inference.py

Expected approximate scores (gpt-4o-mini):
  easy_correctness   -> 0.70-0.85
  medium_performance -> 0.55-0.75
  hard_security      -> 0.45-0.65
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# -- Environment variables ----------------------------------------------------
# As per validator instructions:
#   base_url=os.environ["API_BASE_URL"]
#   api_key=os.environ["API_KEY"]

API_BASE_URL: str = os.environ.get("API_BASE_URL", "").rstrip("/")
API_KEY: str      = os.environ.get("API_KEY", "")
MODEL: str        = os.environ.get("MODEL_NAME", "gpt-4o-mini")

LOCAL_IMAGE_NAME  = os.environ.get("LOCAL_IMAGE_NAME")

# -- Constants ----------------------------------------------------------------
MAX_STEPS               = 6
SUCCESS_SCORE_THRESHOLD = 0.5
BENCHMARK               = "openenv-sql-review"
TASKS                   = ["easy_correctness", "medium_performance", "hard_security"]
TEMPERATURE             = 0.1
MAX_TOKENS              = 1024


# -- Structured log helpers ---------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: Any, reward: float,
             done: bool, error: Optional[str]) -> None:
    error_part = f" error={error}" if error else ""
    print(f"[STEP] step={step} action={action} reward={reward} done={done}{error_part}",
          flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={success} steps={steps} score={score} rewards={rewards}",
          flush=True)


# -- Environment helpers (use API_BASE_URL for /reset /step /grader /health) --

def env_reset(task_id: str, retries: int = 3) -> Dict:
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                f"{API_BASE_URL}/reset",
                params={"task_id": task_id},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            print(f"[DEBUG] env_reset attempt {attempt}/{retries} failed: {exc}", flush=True)
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"env_reset failed after {retries} attempts: {last_exc}")


def env_step(action: Dict, retries: int = 3) -> Dict:
    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                f"{API_BASE_URL}/step",
                json=action,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            print(f"[DEBUG] env_step attempt {attempt}/{retries} failed: {exc}", flush=True)
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"env_step failed after {retries} attempts: {last_exc}")


def env_grade(session_id: str, retries: int = 3) -> float:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                f"{API_BASE_URL}/grader",
                params={"session_id": session_id},
                timeout=30,
            )
            resp.raise_for_status()
            return float(resp.json().get("score", 0.0))
        except Exception as exc:
            print(f"[DEBUG] env_grade attempt {attempt}/{retries} failed: {exc}", flush=True)
            if attempt < retries:
                time.sleep(2 ** attempt)
    print("[DEBUG] env_grade exhausted retries -- returning 0.0", flush=True)
    return 0.0


def wait_for_env(max_wait: int = 120) -> bool:
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
            if resp.status_code == 200:
                print(f"[DEBUG] Environment healthy: {resp.json()}", flush=True)
                return True
        except Exception:
            pass
        print("[DEBUG] Waiting for environment to be ready...", flush=True)
        time.sleep(5)
    return False


# -- LLM agent ----------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior data engineer performing SQL code review.
You will be given a SQL query and schema context. Your job is to:
1. Identify ALL issues: correctness bugs, performance anti-patterns, security vulnerabilities.
2. Rate each issue by severity: critical | high | medium | low.
3. Categorize each issue: correctness | performance | security | style.
4. Suggest a fix where possible.
5. Submit a final decision: approve | request_changes | reject.

DECISION RULES:
- reject:          any critical security or data-integrity issue
- request_changes: high/medium issues but no critical ones
- approve:         only if no real issues found

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
        try:
            cols = ", ".join(
                f"{c['name']} {c['type']}" for c in tbl.get("columns", [])
            )
            idx      = ", ".join(tbl.get("has_index_on", []))
            rows_raw = tbl.get("row_count_hint", "unknown")
            rows_str = f"{rows_raw:,}" if isinstance(rows_raw, int) else str(rows_raw)
            schema_str += (
                f"\nTable: {tbl.get('table_name', '?')} (~{rows_str} rows)\n"
                f"  Columns: {cols}\n"
                f"  Indexes: {idx}\n"
            )
        except Exception as exc:
            print(f"[DEBUG] schema parse error: {exc}", flush=True)

    prior = ""
    for item in obs.get("review_thread", []):
        try:
            if item.get("type") == "issue":
                prior += (
                    f"  - [{item['severity']}] {item['category']}: "
                    f"{item['description']}\n"
                )
        except Exception:
            pass

    return (
        f"Task: {obs.get('task_instructions', '')}\n\n"
        f"=== SQL Query ({obs.get('query_dialect', 'SQL')}) ===\n"
        f"{obs.get('query', '')}\n\n"
        f"=== Schema Context ===\n{schema_str}\n"
        f"=== Issues Already Identified ===\n"
        f"{prior if prior else '  (none yet)'}\n\n"
        "Identify ALL remaining issues and provide your final decision."
    )


def get_model_action(client: OpenAI, obs: Dict) -> Dict:
    fallback: Dict = {
        "issues":   [],
        "comment":  "Unable to generate review.",
        "decision": "request_changes",
    }
    try:
        prompt   = build_user_prompt(obs)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            print("[DEBUG] LLM returned empty content", flush=True)
            return fallback

        if raw.startswith("```"):
            parts = raw.split("```")
            raw   = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected dict, got {type(parsed)}")
        return parsed

    except json.JSONDecodeError as exc:
        print(f"[DEBUG] JSON parse error: {exc}", flush=True)
        return fallback
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return fallback


# -- Per-task episode runner --------------------------------------------------

def run_task(client: OpenAI, task_id: str) -> Dict:
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    session_id:  str         = "unknown"

    log_start(task=task_id, env=BENCHMARK, model=MODEL)

    try:
        obs        = env_reset(task_id)
        session_id = str(obs.get("session_id", "unknown"))
        print(f"[DEBUG] Session: {session_id}", flush=True)

        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            agent_action = get_model_action(client, obs)

            action_payload: Dict = {
                "session_id": session_id,
                "issues":     agent_action.get("issues", []),
                "comment":    agent_action.get("comment"),
                "decision":   agent_action.get("decision"),
            }

            error_msg:    Optional[str] = None
            reward_value: float         = 0.0

            try:
                result     = env_step(action_payload)
                obs        = result.get("observation", obs)
                reward_raw = result.get("reward", 0.0)

                if isinstance(reward_raw, dict):
                    reward_value = float(reward_raw.get("value", 0.0))
                else:
                    reward_value = float(reward_raw)

                done = bool(result.get("done", False))

            except Exception as exc:
                error_msg = str(exc)
                print(f"[DEBUG] env_step error at step {step}: {exc}", flush=True)
                done = True

            rewards.append(reward_value)
            steps_taken = step

            log_step(
                step=step,
                action=action_payload.get("decision", ""),
                reward=reward_value,
                done=done,
                error=error_msg,
            )

            if done:
                break

        if session_id != "unknown":
            score = env_grade(session_id)
        else:
            score = sum(rewards) / max(len(rewards), 1)

        score   = float(min(max(score, 0.0), 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] run_task unhandled error for {task_id!r}: {exc}", flush=True)
        score   = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score":   score,
        "steps":   steps_taken,
        "success": success,
        "rewards": rewards,
    }


# -- Main ---------------------------------------------------------------------

def main() -> None:
    if not API_BASE_URL:
        print("[warn] API_BASE_URL is not set -- all calls will fail.", flush=True)
    if not API_KEY:
        print("[warn] API_KEY is not set -- LLM calls will fail.", flush=True)
    if not os.environ.get("MODEL_NAME"):
        print(f"[warn] MODEL_NAME not set, defaulting to {MODEL!r}", flush=True)

    print(f"[DEBUG] API_BASE_URL = {API_BASE_URL!r}", flush=True)
    print(f"[DEBUG] MODEL        = {MODEL!r}", flush=True)

    if not wait_for_env(max_wait=120):
        print(
            f"[warn] Environment at {API_BASE_URL!r} not healthy after 120s -- "
            "proceeding anyway.", flush=True
        )

    # Exactly as the validator instructs:
    # base_url=os.environ["API_BASE_URL"], api_key=os.environ["API_KEY"]
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_results: List[Dict] = []
    for task_id in TASKS:
        try:
            result = run_task(client, task_id)
            all_results.append(result)
        except Exception as exc:
            print(f"[DEBUG] Unexpected error running task {task_id!r}: {exc}", flush=True)
            log_end(success=False, steps=0, score=0.0, rewards=[])
            all_results.append({
                "task_id": task_id,
                "score":   0.0,
                "steps":   0,
                "success": False,
                "rewards": [],
            })
        time.sleep(1)

    sep = "=" * 60
    print(f"\n{sep}", flush=True)
    print("BASELINE RESULTS", flush=True)
    print(sep, flush=True)
    for r in all_results:
        print(f"  {r['task_id']:30s}  score={r['score']:.4f}", flush=True)

    avg = (
        sum(r["score"] for r in all_results) / len(all_results)
        if all_results else 0.0
    )
    print(f"\n  Average score: {avg:.4f}", flush=True)
    print(f"{sep}\n", flush=True)

    try:
        with open("baseline_results.json", "w") as fh:
            json.dump({"results": all_results, "average": avg}, fh, indent=2)
        print("Results written to baseline_results.json", flush=True)
    except Exception as exc:
        print(f"[DEBUG] Could not write baseline_results.json: {exc}", flush=True)


if __name__ == "__main__":
    main()