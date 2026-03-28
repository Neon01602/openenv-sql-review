"""
SQL Query Review Environment — OpenEnv compliant FastAPI application.

Domain: SQL code review. An agent acts as a senior data engineer reviewing
submitted SQL queries for correctness, performance anti-patterns, and
security vulnerabilities (SQL injection risks).

This simulates a genuine daily workflow in data engineering teams.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .models import Action, Observation, Reward, TaskInfo
from .session import SessionManager
from .tasks import TASK_REGISTRY

app = FastAPI(
    title="SQL Query Review — OpenEnv",
    description="An OpenEnv environment that trains agents to review SQL queries for correctness, performance, and security.",
    version="1.0.0",
)

sessions = SessionManager()


# ─────────────────────────────────────────────
# Core OpenEnv endpoints
# ─────────────────────────────────────────────


@app.post("/reset", response_model=Observation)
def reset(task_id: Optional[str] = None):
    """Start a new episode. Returns the initial observation."""
    if task_id is None:
        task_id = "easy_correctness"
    if task_id not in TASK_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")

    task = TASK_REGISTRY[task_id]
    session_id = str(uuid.uuid4())
    obs = sessions.new_session(session_id, task)
    return obs


@app.post("/step", response_model=dict)
def step(action: Action):
    """
    Apply an agent action to the current episode.
    Returns observation, reward, done, info.
    """
    session = sessions.get(action.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")

    obs, reward, done, info = session.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(session_id: str):
    """Return the full current state of an episode."""
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session.get_state()


# ─────────────────────────────────────────────
# Required extra endpoints
# ─────────────────────────────────────────────


@app.get("/tasks")
def list_tasks():
    """Return all tasks and the action schema."""
    return {
        "tasks": [
            {
                "id": tid,
                "name": t.name,
                "difficulty": t.difficulty,
                "description": t.description,
                "max_steps": t.max_steps,
            }
            for tid, t in TASK_REGISTRY.items()
        ],
        "action_schema": Action.model_json_schema(),
    }


@app.get("/grader")
def grader(session_id: str):
    """Return the grader score for a completed episode."""
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    score = session.grade()
    return {"session_id": session_id, "score": score, "task_id": session.task.task_id}


@app.post("/baseline")
def baseline():
    """
    Run the built-in heuristic baseline agent across all 3 tasks.
    Returns reproducible scores (no LLM needed for the baseline gate).
    """
    from .baseline import run_heuristic_baseline

    results = run_heuristic_baseline()
    return {"baseline_results": results}


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/")
def root():
    return {
        "environment": "SQL Query Review",
        "version": "1.0.0",
        "spec": "openenv-v1",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline", "/health"],
    }
