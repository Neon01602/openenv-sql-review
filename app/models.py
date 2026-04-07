"""
Pydantic models for the SQL Query Review OpenEnv environment.
Observation, Action, Reward — all typed per OpenEnv spec.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class IssueCategory(str, Enum):
    correctness = "correctness"
    performance = "performance"
    security    = "security"
    style       = "style"

class IssueSeverity(str, Enum):
    critical = "critical"
    high     = "high"
    medium   = "medium"
    low      = "low"

class ReviewDecision(str, Enum):
    approve         = "approve"
    request_changes = "request_changes"
    reject          = "reject"


# ── Sub-models ───────────────────────────────────────────────────────────────

class ColumnInfo(BaseModel):
    name: str
    type: str
    nullable: bool = True

class SchemaInfo(BaseModel):
    table_name: str
    columns: list[ColumnInfo]
    row_count_hint: int = 0
    has_index_on: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)  # e.g. FK, UNIQUE

class ReviewIssue(BaseModel):
    category:     IssueCategory
    severity:     IssueSeverity
    line_hint:    Optional[int]   = None
    description:  str
    suggested_fix: Optional[str] = None


# ── Observation ──────────────────────────────────────────────────────────────

class Observation(BaseModel):
    session_id:        str
    task_id:           str
    step_number:       int
    max_steps:         int
    query:             str
    query_dialect:     str = "PostgreSQL"
    schema_context:    list[SchemaInfo]
    review_thread:     list[dict[str, Any]] = Field(default_factory=list)
    task_instructions: str
    done:              bool = False


# ── TaskInfo ────────────────────────────────────────────────────────────────

class TaskInfo(BaseModel):
    task_id: str
    name: str
    difficulty: str
    description: str
    max_steps: int

# ── Action ───────────────────────────────────────────────────────────────────

class Action(BaseModel):
    session_id: str
    issues:     list[ReviewIssue] = Field(default_factory=list)
    comment:    Optional[str]     = None
    decision:   Optional[ReviewDecision] = None


# ── Reward ───────────────────────────────────────────────────────────────────

class Reward(BaseModel):
    value:                  float
    cumulative:             float
    issue_detection_reward: float = 0.0
    false_positive_penalty: float = 0.0
    severity_accuracy_reward: float = 0.0
    decision_reward:        float = 0.0
    step_efficiency_bonus:  float = 0.0
    message:                str   = ""


# ── Grader response ──────────────────────────────────────────────────────────

class GraderResult(BaseModel):
    session_id: str
    task_id:    str
    score:      float   # 0.0 – 1.0
    breakdown:  dict[str, Any] = Field(default_factory=dict)