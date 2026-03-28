"""
SQL Query Review — OpenEnv Environment
FastAPI application exposing all required OpenEnv endpoints.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.models import Action, GraderResult, Observation, Reward
from app.session import create_episode, get_episode
from app.tasks import ACTION_SCHEMA, list_tasks, get_task, generate_task

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "SQL Query Review — OpenEnv",
    description = (
        "An OpenEnv-compliant reinforcement-learning environment that trains and "
        "evaluates AI agents to perform SQL code review. "
        "Agents review PostgreSQL queries for correctness bugs, performance "
        "anti-patterns, and security vulnerabilities.\n\n"
        "**6 hand-crafted tasks** (easy × 2, medium × 2, hard × 2) plus "
        "**procedurally generated tasks** via `generated_<difficulty>_<seed>`."
    ),
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Core OpenEnv endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post(
    "/reset",
    response_model = Observation,
    summary        = "Reset",
    description    = (
        "Start a new episode. Returns the initial observation.\n\n"
        "Pass one of the 6 fixed task IDs or a procedurally generated task like "
        "`generated_hard_42`."
    ),
)
def reset(task_id: Optional[str] = Query(default="easy_correctness")) -> Observation:
    try:
        ep = create_episode(task_id or "easy_correctness")
    except KeyError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return ep.initial_observation()


@app.post(
    "/step",
    summary     = "Step",
    description = "Apply an agent action to the current episode. Returns observation, reward, done, info.",
)
def step(action: Action) -> dict[str, Any]:
    try:
        ep = get_episode(action.session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        obs, reward, done, info = ep.step(action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "observation": obs.model_dump(),
        "reward":      reward.model_dump(),
        "done":        done,
        "info":        info,
    }


@app.get(
    "/state",
    summary     = "State",
    description = "Return the full current state of an episode.",
)
def state(session_id: str = Query(...)) -> dict[str, Any]:
    try:
        ep = get_episode(session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ep.state()


@app.get(
    "/grader",
    response_model = GraderResult,
    summary        = "Grader",
    description    = "Return the grader score for a completed episode.",
)
def grader(session_id: str = Query(...)) -> GraderResult:
    try:
        ep = get_episode(session_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ep.grade()


@app.get(
    "/tasks",
    summary     = "List Tasks",
    description = "Return the list of available tasks and the action schema.",
)
def tasks() -> dict[str, Any]:
    return {
        "tasks":         list_tasks(),
        "action_schema": ACTION_SCHEMA,
        "procedural_tasks": {
            "description": (
                "Generate novel tasks by passing task_id=generated_<difficulty>_<seed> "
                "to /reset. Example: generated_hard_42"
            ),
            "difficulties": ["easy", "medium", "hard"],
            "example_ids":  [f"generated_{d}_{i}" for d, i in
                             [("easy",7), ("medium",42), ("hard",99)]],
        },
    }


@app.post(
    "/baseline",
    summary     = "Baseline",
    description = (
        "Run the heuristic baseline agent against all 6 fixed tasks. "
        "No API key required. Scores are deterministic and reproducible."
    ),
)
def baseline() -> dict[str, Any]:
    from app.baseline import run_heuristic_baseline
    results = run_heuristic_baseline()
    avg = round(sum(r["score"] for r in results) / len(results), 4)
    return {
        "results": results,
        "average_score": avg,
        "agent": "heuristic_regex_v2",
        "note": "Deterministic, no LLM required.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Utility endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health", description="Liveness probe.")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "2.0.0"}


@app.get(
    "/",
    summary     = "Root",
    description = "Environment metadata.",
)
def root() -> dict[str, Any]:
    return {
        "name":        "SQL Query Review",
        "version":     "2.0.0",
        "spec":        "openenv-v1",
        "description": (
            "An RL environment for training and evaluating agents on SQL code review. "
            "6 fixed tasks (easy → hard) + procedurally generated tasks."
        ),
        "tasks":       [t["id"] for t in list_tasks()],
        "endpoints":   ["/reset", "/step", "/state", "/grader", "/tasks", "/baseline", "/health"],
        "docs":        "/docs",
    }