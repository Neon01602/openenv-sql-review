"""
SQL Query Review Environment — OpenEnv compliant FastAPI application.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import Action, Observation, Reward, TaskInfo
from .session import SessionManager
from .tasks import TASK_REGISTRY

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

sessions = SessionManager()


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
    """Apply an agent action to the current episode."""
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
    """Run the built-in heuristic baseline agent across all tasks."""
    from .baseline import run_heuristic_baseline
    results = run_heuristic_baseline()
    return {"baseline_results": results}


@app.get("/health")
def health():
    # openenv validate requires "healthy" not "ok"
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
        "tasks": list(TASK_REGISTRY.keys()),
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
                "session_id":        {"type": "string"},
                "task_id":           {"type": "string"},
                "step_number":       {"type": "integer"},
                "max_steps":         {"type": "integer"},
                "done":              {"type": "boolean"},
                "cumulative_reward": {"type": "number"},
                "agent_issues":      {"type": "array"},
                "agent_decision":    {"type": ["string", "null"]},
                "grade":             {"type": ["number", "null"]},
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
                {"name": "reset",  "description": "Start a new episode"},
                {"name": "step",   "description": "Apply an agent action"},
                {"name": "state",  "description": "Get current episode state"},
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
