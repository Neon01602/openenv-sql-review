"""
SQL Query Review Environment — OpenEnv compliant FastAPI application.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import Action, Observation, Reward
from .session import create_episode, get_episode
from .tasks import TASK_MAP, list_tasks as _list_tasks, ACTION_SCHEMA

app = FastAPI(
    title="SQL Query Review — OpenEnv",
    description="An OpenEnv environment that trains agents to review SQL queries for correctness, performance, and security.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/reset", response_model=Observation)
def reset(task_id: Optional[str] = None):
    """Start a new episode. Returns the initial observation."""
    if task_id is None:
        task_id = "easy_correctness"
    if task_id not in TASK_MAP and not task_id.startswith("generated_"):
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    ep = create_episode(task_id)
    return ep.initial_observation()


@app.post("/step", response_model=dict)
def step(action: Action):
    """Apply an agent action to the current episode."""
    try:
        ep = get_episode(action.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")
    obs, reward, done, info = ep.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(session_id: str):
    """Return the full current state of an episode."""
    try:
        ep = get_episode(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found.")
    return ep.state()


@app.get("/tasks")
def list_tasks_endpoint():
    """Return all tasks and the action schema."""
    return {
        "tasks": _list_tasks(),
        "action_schema": ACTION_SCHEMA,
    }


@app.get("/grader")
def grader(session_id: str):
    """Return the grader score for a completed episode."""
    try:
        ep = get_episode(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found.")
    result = ep.grade()
    return result.model_dump()


@app.post("/baseline")
def baseline():
    """Run the built-in heuristic baseline agent across all tasks."""
    from .baseline import run_heuristic_baseline
    results = run_heuristic_baseline()
    return {"baseline_results": results}


@app.get("/health")
def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/metadata")
def metadata():
    """Environment metadata — required by openenv validate."""
    return {
        "name": "SQL Query Review",
        "description": (
            "An OpenEnv environment that trains agents to review SQL queries "
            "for correctness bugs, performance anti-patterns, and security vulnerabilities."
        ),
        "version": "1.0.0",
        "spec": "openenv-v1",
        "tasks": list(TASK_MAP.keys()),
        "endpoints": [
            "/reset", "/step", "/state", "/tasks", "/grader",
            "/baseline", "/health", "/metadata", "/schema", "/mcp",
        ],
        "docs": "/docs",
    }


@app.get("/schema")
def schema():
    """Action, observation, and state schemas — required by openenv validate."""
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "task_id": {"type": "string"},
                "step_number": {"type": "integer"},
                "max_steps": {"type": "integer"},
                "done": {"type": "boolean"},
                "cumulative_reward": {"type": "number"},
                "gt_detected": {"type": "integer"},
                "gt_total": {"type": "integer"},
                "false_positives": {"type": "integer"},
                "final_decision": {"type": ["string", "null"]},
            },
        },
    }


@app.post("/mcp")
def mcp(request: dict = {}):
    """JSON-RPC 2.0 endpoint — required by openenv validate."""
    return {
        "jsonrpc": "2.0",
        "id": request.get("id", 1),
        "result": {
            "tools": [
                {"name": "reset", "description": "Start a new episode"},
                {"name": "step", "description": "Apply an agent action"},
                {"name": "state", "description": "Get current episode state"},
                {"name": "grader", "description": "Get final episode score"},
            ]
        },
    }


@app.get("/")
def root():
    return {
        "environment": "SQL Query Review",
        "version": "1.0.0",
        "spec": "openenv-v1",
        "endpoints": [
            "/reset", "/step", "/state", "/tasks", "/grader",
            "/baseline", "/health", "/metadata", "/schema", "/mcp",
        ],
    }