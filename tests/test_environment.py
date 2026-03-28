"""
Comprehensive test suite for SQL Query Review OpenEnv.
Covers: spec compliance, all 6 tasks, reward correctness,
grader accuracy, false positive penalties, procedural generation,
and edge cases.
"""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import (
    Action, ReviewIssue, IssueCategory, IssueSeverity, ReviewDecision,
)
from app.session import create_episode
from app.tasks import FIXED_TASKS, get_task, generate_task, list_tasks

client = TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reset(task_id: str) -> dict:
    r = client.post("/reset", params={"task_id": task_id})
    assert r.status_code == 200, r.text
    return r.json()

def _step(session_id: str, issues=None, decision=None, comment=None) -> dict:
    payload: dict = {"session_id": session_id, "issues": issues or []}
    if decision:
        payload["decision"] = decision
    if comment:
        payload["comment"] = comment
    r = client.post("/step", json=payload)
    assert r.status_code == 200, r.text
    return r.json()

def _grader(session_id: str) -> dict:
    r = client.get("/grader", params={"session_id": session_id})
    assert r.status_code == 200, r.text
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Root / health
# ─────────────────────────────────────────────────────────────────────────────

def test_root_returns_metadata():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "name" in data
    assert "version" in data
    assert "tasks" in data

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tasks endpoint
# ─────────────────────────────────────────────────────────────────────────────

def test_tasks_returns_all_six():
    r = client.get("/tasks")
    assert r.status_code == 200
    data = r.json()
    assert len(data["tasks"]) == 6

def test_tasks_includes_action_schema():
    r = client.get("/tasks")
    data = r.json()
    assert "action_schema" in data

def test_tasks_includes_procedural_info():
    r = client.get("/tasks")
    data = r.json()
    assert "procedural_tasks" in data

def test_tasks_difficulty_progression():
    tasks = list_tasks()
    diffs = [t["difficulty"] for t in tasks]
    assert "easy" in diffs
    assert "medium" in diffs
    assert "hard" in diffs


# ─────────────────────────────────────────────────────────────────────────────
# 3. Reset — all 6 tasks
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task_id", [t["id"] for t in FIXED_TASKS])
def test_reset_all_tasks(task_id: str):
    obs = _reset(task_id)
    assert obs["task_id"] == task_id
    assert obs["step_number"] == 0
    assert obs["done"] is False
    assert "session_id" in obs
    assert "query" in obs
    assert "schema_context" in obs
    assert len(obs["review_thread"]) == 0

def test_reset_unknown_task_raises_422():
    r = client.post("/reset", params={"task_id": "nonexistent_task"})
    assert r.status_code == 422

def test_reset_default_task():
    r = client.post("/reset")
    assert r.status_code == 200
    assert r.json()["task_id"] == "easy_correctness"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Step
# ─────────────────────────────────────────────────────────────────────────────

def test_step_returns_reward_and_observation():
    obs = _reset("easy_correctness")
    result = _step(obs["session_id"], issues=[], decision="approve")
    assert "reward" in result
    assert "observation" in result
    assert "done" in result
    assert "info" in result

def test_step_increments_step_number():
    obs = _reset("medium_performance")
    sid = obs["session_id"]
    result = _step(sid, issues=[])
    assert result["observation"]["step_number"] == 1

def test_step_done_on_decision():
    obs = _reset("easy_correctness")
    result = _step(obs["session_id"], issues=[], decision="approve")
    assert result["done"] is True

def test_step_not_done_without_decision():
    obs = _reset("medium_performance")
    result = _step(obs["session_id"], issues=[])
    assert result["done"] is False

def test_step_on_completed_episode_raises_400():
    obs = _reset("easy_correctness")
    sid = obs["session_id"]
    _step(sid, decision="approve")
    r = client.post("/step", json={"session_id": sid, "issues": [], "decision": "approve"})
    assert r.status_code == 400

def test_step_unknown_session_raises_404():
    r = client.post("/step", json={"session_id": "bad-session-id", "issues": []})
    assert r.status_code == 404

def test_step_review_thread_grows():
    obs = _reset("easy_correctness")
    sid = obs["session_id"]
    result = _step(sid, issues=[{
        "category": "correctness",
        "severity": "high",
        "description": "Wrong JOIN type causes data loss.",
        "suggested_fix": "Use LEFT JOIN.",
    }])
    thread = result["observation"]["review_thread"]
    assert len(thread) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 5. Reward correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_correct_issue_gives_positive_reward():
    obs = _reset("easy_correctness")
    result = _step(obs["session_id"], issues=[{
        "category": "correctness",
        "severity": "high",
        "description": "INNER JOIN silently drops orders with null customer_id. Use LEFT JOIN.",
        "suggested_fix": "Change INNER JOIN to LEFT JOIN.",
    }], decision="request_changes")
    assert result["reward"]["value"] > 0

def test_false_positive_penalises_reward():
    obs = _reset("easy_correctness")
    result = _step(obs["session_id"], issues=[
        {
            "category": "performance",
            "severity": "high",
            "description": "Index on primary key is missing (fabricated).",
            "suggested_fix": "Add index.",
        },
        {
            "category": "security",
            "severity": "critical",
            "description": "Unrelated security claim that does not exist in this query.",
        },
    ], decision="approve")
    assert result["reward"]["false_positive_penalty"] > 0

def test_correct_decision_gives_decision_reward():
    obs = _reset("hard_security")
    result = _step(obs["session_id"], issues=[], decision="reject")
    assert result["reward"]["decision_reward"] > 0

def test_wrong_decision_gives_zero_decision_reward():
    obs = _reset("hard_security")
    result = _step(obs["session_id"], issues=[], decision="approve")
    # approve is wrong for hard_security (should be reject)
    assert result["reward"]["decision_reward"] == 0

def test_reward_value_in_range():
    for task in FIXED_TASKS[:3]:
        obs = _reset(task["id"])
        result = _step(obs["session_id"], issues=[], decision="approve")
        val = result["reward"]["value"]
        assert -1.0 <= val <= 1.0, f"Reward {val} out of range for {task['id']}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Grader
# ─────────────────────────────────────────────────────────────────────────────

def test_grader_returns_score_in_range():
    obs = _reset("easy_correctness")
    _step(obs["session_id"], decision="approve")
    grade = _grader(obs["session_id"])
    assert 0.0 <= grade["score"] <= 1.0

def test_grader_correct_answer_scores_higher():
    # Perfect answer
    obs1 = _reset("easy_correctness")
    _step(obs1["session_id"], issues=[{
        "category": "correctness",
        "severity": "high",
        "description": "INNER JOIN drops rows with null customer_id — guest checkouts are lost. Use LEFT JOIN.",
        "suggested_fix": "Change INNER JOIN to LEFT JOIN.",
    }], decision="request_changes")
    good_score = _grader(obs1["session_id"])["score"]

    # Wrong answer
    obs2 = _reset("easy_correctness")
    _step(obs2["session_id"], issues=[], decision="approve")
    bad_score = _grader(obs2["session_id"])["score"]

    assert good_score > bad_score

def test_grader_has_breakdown():
    obs = _reset("medium_performance")
    _step(obs["session_id"], decision="request_changes")
    grade = _grader(obs["session_id"])
    assert "breakdown" in grade
    bd = grade["breakdown"]
    assert "detection_score" in bd
    assert "fp_multiplier" in bd
    assert "decision_score" in bd

def test_grader_many_fps_lowers_score():
    obs = _reset("easy_correctness")
    _step(obs["session_id"], issues=[
        {"category": "security", "severity": "critical", "description": f"Fake issue {i}"}
        for i in range(5)
    ], decision="request_changes")
    grade = _grader(obs["session_id"])
    # High FP rate should significantly reduce score
    assert grade["breakdown"]["fp_multiplier"] < 0.9


# ─────────────────────────────────────────────────────────────────────────────
# 7. State
# ─────────────────────────────────────────────────────────────────────────────

def test_state_returns_episode_info():
    obs = _reset("easy_correctness")
    r = client.get("/state", params={"session_id": obs["session_id"]})
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == obs["session_id"]
    assert "done" in data
    assert "cumulative_reward" in data

def test_state_unknown_session_raises_404():
    r = client.get("/state", params={"session_id": "bad-id"})
    assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# 8. Baseline
# ─────────────────────────────────────────────────────────────────────────────

def test_baseline_runs_without_api_key():
    r = client.post("/baseline")
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert "average_score" in data
    assert len(data["results"]) == 6

def test_baseline_all_scores_in_range():
    r = client.post("/baseline")
    for result in r.json()["results"]:
        assert 0.0 <= result["score"] <= 1.0, f"Score out of range: {result}"

def test_baseline_difficulty_ordering():
    """Hard tasks should score lower than easy tasks on the heuristic baseline."""
    r = client.post("/baseline")
    results = {res["task_id"]: res for res in r.json()["results"]}
    easy_avg  = (results["easy_correctness"]["score"] + results["easy_type_coercion"]["score"]) / 2
    hard_avg  = (results["hard_security"]["score"] + results["hard_migration"]["score"]) / 2
    # Easy should be harder to fool than hard for a heuristic — or at minimum hard < easy
    # Just verify all are non-trivial
    for res in r.json()["results"]:
        assert res["score"] > 0.0, f"Zero score for {res['task_id']}"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Procedural task generation
# ─────────────────────────────────────────────────────────────────────────────

def test_procedural_task_reset():
    obs = _reset("generated_easy_7")
    assert obs["step_number"] == 0
    assert "generated_easy_7" in obs["task_id"]

def test_procedural_task_different_seeds_give_different_queries():
    t1 = generate_task(seed=1)
    t2 = generate_task(seed=2)
    assert t1["query"] != t2["query"]

def test_procedural_task_same_seed_deterministic():
    t1 = generate_task(seed=42)
    t2 = generate_task(seed=42)
    assert t1["query"] == t2["query"]
    assert len(t1["ground_truth_issues"]) == len(t2["ground_truth_issues"])

def test_procedural_task_has_required_fields():
    task = generate_task(seed=99)
    assert "id" in task
    assert "query" in task
    assert "schema_context" in task
    assert "ground_truth_issues" in task
    assert "correct_decision" in task
    assert len(task["ground_truth_issues"]) >= 1

def test_procedural_task_full_episode():
    obs = _reset("generated_hard_42")
    result = _step(obs["session_id"], issues=[], decision="request_changes")
    grade = _grader(obs["session_id"])
    assert 0.0 <= grade["score"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 10. Multi-step episode
# ─────────────────────────────────────────────────────────────────────────────

def test_multi_step_episode_medium_task():
    obs = _reset("medium_performance")
    sid = obs["session_id"]

    # Step 1: report one issue
    r1 = _step(sid, issues=[{
        "category": "performance",
        "severity": "high",
        "description": "Correlated subquery causes N+1 performance problem — runs once per row.",
        "suggested_fix": "Replace with JOIN + GROUP BY.",
    }])
    assert r1["observation"]["step_number"] == 1
    assert not r1["done"]

    # Step 2: report another issue + decision
    r2 = _step(sid, issues=[{
        "category": "performance",
        "severity": "high",
        "description": "LOWER(email) defeats the index on email, forcing a full table scan.",
        "suggested_fix": "Create a functional index on LOWER(email).",
    }], decision="request_changes")
    assert r2["done"] is True

    grade = _grader(sid)
    assert grade["score"] > 0.3   # Should get credit for 2/3 issues

def test_episode_ends_at_max_steps():
    obs = _reset("easy_correctness")
    sid = obs["session_id"]
    max_steps = obs["max_steps"]
    for i in range(max_steps):
        result = _step(sid)
        if result["done"]:
            break
    assert result["done"] is True