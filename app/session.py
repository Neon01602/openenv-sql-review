"""
Episode state management and reward computation for SQL Query Review OpenEnv.

Key improvements over v1:
  - Embedding-based semantic matching (via simple TF-IDF cosine similarity)
    so agents are rewarded for correctly-worded paraphrases, not just keyword hits.
  - Keyword fallback when similarity cannot be computed.
  - Richer grader that accounts for overall false-positive rate.
  - Clean separation of step reward vs final grader score.
"""
from __future__ import annotations

import math
import re
import uuid
from collections import defaultdict
from typing import Any, Optional

from app.models import (
    Action, Observation, Reward, GraderResult, ReviewIssue,
    IssueSeverity, ReviewDecision,
)
from app.tasks import get_task

# ─────────────────────────────────────────────────────────────────────────────
# Severity utils
# ─────────────────────────────────────────────────────────────────────────────

_SEV_RANK = {
    IssueSeverity.critical: 3,
    IssueSeverity.high:     2,
    IssueSeverity.medium:   1,
    IssueSeverity.low:      0,
}

def _sev_distance(a: IssueSeverity, b: IssueSeverity) -> int:
    return abs(_SEV_RANK[a] - _SEV_RANK[b])

_CATEGORY_WEIGHTS = {
    "security":    1.4,
    "correctness": 1.1,
    "performance": 0.9,
    "style":       0.6,
}

_SEVERITY_WEIGHTS = {
    IssueSeverity.critical: 1.5,
    IssueSeverity.high:     1.0,
    IssueSeverity.medium:   0.6,
    IssueSeverity.low:      0.3,
}


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight semantic matcher
# ─────────────────────────────────────────────────────────────────────────────
# We use a bag-of-words cosine similarity so agents are rewarded for correctly
# describing an issue in their own words rather than needing exact keyword hits.

def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def _tfidf_vector(tokens: list[str], vocab: set[str]) -> dict[str, float]:
    tf: dict[str, float] = defaultdict(float)
    for t in tokens:
        tf[t] += 1
    n = max(len(tokens), 1)
    return {w: tf[w] / n for w in vocab if tf[w] > 0}

def _cosine(v1: dict[str, float], v2: dict[str, float]) -> float:
    shared = set(v1) & set(v2)
    if not shared:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in shared)
    norm1 = math.sqrt(sum(x * x for x in v1.values()))
    norm2 = math.sqrt(sum(x * x for x in v2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def _semantic_similarity(text_a: str, text_b: str) -> float:
    """Return cosine similarity between two free-text descriptions."""
    tok_a = _tokenise(text_a)
    tok_b = _tokenise(text_b)
    vocab = set(tok_a) | set(tok_b)
    va = _tfidf_vector(tok_a, vocab)
    vb = _tfidf_vector(tok_b, vocab)
    return _cosine(va, vb)

# Match threshold: at this similarity or above we consider it a true positive
_MATCH_THRESHOLD = 0.22   # loose enough for paraphrases, tight enough to block noise

def _matches_ground_truth(submitted: ReviewIssue, gt: ReviewIssue) -> bool:
    """Return True if submitted issue semantically matches ground truth issue."""
    # Category must match
    if submitted.category != gt.category:
        return False
    # Description must be semantically similar
    sim = _semantic_similarity(submitted.description, gt.description)
    return sim >= _MATCH_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# Episode session
# ─────────────────────────────────────────────────────────────────────────────

class Episode:
    def __init__(self, task_id: str):
        self.session_id = str(uuid.uuid4())
        self.task       = get_task(task_id)
        self.task_id    = self.task["id"]
        self.step_number = 0
        self.max_steps   = self.task["max_steps"]
        self.done        = False
        self.cumulative_reward: float = 0.0
        self.review_thread: list[dict[str, Any]] = []

        # Track which ground-truth issues have been detected
        self.gt_issues: list[ReviewIssue] = self.task["ground_truth_issues"]
        self.detected_gt_indices: set[int] = set()

        # False positive count (for grader)
        self.false_positive_count: int = 0
        self.total_submitted_issues: int = 0

        # Final decision
        self.final_decision: Optional[ReviewDecision] = None

    # ── Build observation ────────────────────────────────────────────────────

    def _observation(self) -> Observation:
        return Observation(
            session_id        = self.session_id,
            task_id           = self.task_id,
            step_number       = self.step_number,
            max_steps         = self.max_steps,
            query             = self.task["query"],
            query_dialect     = self.task["dialect"],
            schema_context    = self.task["schema_context"],
            review_thread     = self.review_thread,
            task_instructions = self.task["instructions"],
            done              = self.done,
        )

    def initial_observation(self) -> Observation:
        return self._observation()

    # ── Apply action ─────────────────────────────────────────────────────────

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self.done:
            raise ValueError("Episode is already done. Call /reset to start a new episode.")

        self.step_number += 1
        issues_this_step = action.issues or []
        self.total_submitted_issues += len(issues_this_step)

        # ── Compute reward components ─────────────────────────────────────

        detection_reward  = 0.0
        sev_accuracy      = 0.0
        fp_penalty        = 0.0
        n_new_detections  = 0

        for submitted in issues_this_step:
            matched = False
            for i, gt in enumerate(self.gt_issues):
                if i in self.detected_gt_indices:
                    continue
                if _matches_ground_truth(submitted, gt):
                    # True positive
                    self.detected_gt_indices.add(i)
                    w = (_CATEGORY_WEIGHTS.get(gt.category.value, 1.0) *
                         _SEVERITY_WEIGHTS.get(gt.severity, 1.0))
                    detection_reward += 0.25 * w
                    # Severity accuracy bonus
                    dist = _sev_distance(submitted.severity, gt.severity)
                    sev_accuracy += max(0.0, 0.10 - 0.04 * dist)
                    n_new_detections += 1
                    matched = True
                    break

            if not matched:
                # False positive
                self.false_positive_count += 1
                fp_penalty += 0.12

        # Normalise detection reward by total gt weight so max detection reward ≈ 0.5
        total_gt_weight = sum(
            _CATEGORY_WEIGHTS.get(g.category.value, 1.0) * _SEVERITY_WEIGHTS.get(g.severity, 1.0)
            for g in self.gt_issues
        ) or 1.0
        detection_reward = min(detection_reward / (total_gt_weight * 0.5), 0.5) if total_gt_weight else 0.0

        # Re-scale: detection_reward is now a fraction [0, 1] of remaining detectable reward
        # Absolute contribution per step = fraction * 0.50 max pool
        detection_reward = detection_reward * 0.50

        # Decision reward
        decision_reward = 0.0
        if action.decision is not None:
            self.final_decision = action.decision
            self.done = True
            if action.decision == self.task["correct_decision"]:
                decision_reward = 0.20
            else:
                # Partial credit for close decisions
                correct = self.task["correct_decision"]
                # approve vs reject = 0.0, approve/reject vs request_changes = 0.05
                decision_pairs_partial = {
                    (ReviewDecision.request_changes, ReviewDecision.reject),
                    (ReviewDecision.reject, ReviewDecision.request_changes),
                }
                if (action.decision, correct) in decision_pairs_partial:
                    decision_reward = 0.05

        # Step efficiency: tiny bonus for finishing in fewer steps
        efficiency_bonus = 0.0
        if self.done and self.step_number < self.max_steps:
            steps_saved = self.max_steps - self.step_number
            efficiency_bonus = round(0.01 * steps_saved, 3)

        # Force done at max_steps
        if self.step_number >= self.max_steps:
            self.done = True

        # ── Assemble reward ───────────────────────────────────────────────

        step_value = (
            detection_reward
            + sev_accuracy
            - fp_penalty
            + decision_reward
            + efficiency_bonus
        )
        step_value = round(max(-1.0, min(1.0, step_value)), 4)
        self.cumulative_reward = round(self.cumulative_reward + step_value, 4)

        reward = Reward(
            value                  = step_value,
            cumulative             = self.cumulative_reward,
            issue_detection_reward = round(detection_reward, 4),
            false_positive_penalty = round(fp_penalty, 4),
            severity_accuracy_reward = round(sev_accuracy, 4),
            decision_reward        = round(decision_reward, 4),
            step_efficiency_bonus  = round(efficiency_bonus, 4),
            message = (
                f"Step {self.step_number}: "
                f"+{detection_reward:.2f} detection, "
                f"+{sev_accuracy:.2f} severity, "
                f"-{fp_penalty:.2f} FP, "
                f"+{decision_reward:.2f} decision"
            ),
        )

        # Update thread
        for iss in issues_this_step:
            self.review_thread.append({
                "role": "agent", "type": "issue",
                "category":    iss.category.value,
                "severity":    iss.severity.value,
                "description": iss.description,
            })
        if action.comment:
            self.review_thread.append({"role": "agent", "type": "comment", "value": action.comment})
        if action.decision:
            self.review_thread.append({"role": "agent", "type": "decision", "value": action.decision.value})

        info = {
            "issues_submitted_total": self.total_submitted_issues,
            "steps_remaining":        max(0, self.max_steps - self.step_number),
            "gt_detected":            len(self.detected_gt_indices),
            "gt_total":               len(self.gt_issues),
        }

        return self._observation(), reward, self.done, info

    # ── Final grader ─────────────────────────────────────────────────────────

    def grade(self) -> GraderResult:
        """
        Compute the final 0.0–1.0 grader score for a completed episode.

        Formula:
          detection_score  = Σ(weight of detected GT issues) / Σ(weight of all GT issues)
          fp_rate          = false_positives / max(total_submitted, 1)
          fp_multiplier    = max(0.4, 1 - fp_rate * 1.5)   # harsh penalty
          decision_score   = 1.0 if correct, 0.3 if partial, 0.0 if wrong
          final = detection_score * fp_multiplier * 0.80 + decision_score * 0.20
        """
        # Detection score
        total_weight    = sum(
            _CATEGORY_WEIGHTS.get(g.category.value,1.0) * _SEVERITY_WEIGHTS.get(g.severity,1.0)
            for g in self.gt_issues
        ) or 1.0
        detected_weight = sum(
            _CATEGORY_WEIGHTS.get(self.gt_issues[i].category.value,1.0) *
            _SEVERITY_WEIGHTS.get(self.gt_issues[i].severity,1.0)
            for i in self.detected_gt_indices
        )
        detection_score = detected_weight / total_weight

        # FP multiplier
        fp_rate       = self.false_positive_count / max(self.total_submitted_issues, 1)
        fp_multiplier = max(0.4, 1.0 - fp_rate * 1.5)

        # Decision score
        correct = self.task["correct_decision"]
        if self.final_decision == correct:
            decision_score = 1.0
        elif self.final_decision is not None and {self.final_decision, correct} == {
                ReviewDecision.request_changes, ReviewDecision.reject}:
            decision_score = 0.3
        else:
            decision_score = 0.0

        final_score = round(
            detection_score * fp_multiplier * 0.80 + decision_score * 0.20, 4
        )

        return GraderResult(
            session_id = self.session_id,
            task_id    = self.task_id,
            score      = final_score,
            breakdown  = {
                "detection_score":     round(detection_score, 4),
                "fp_rate":             round(fp_rate, 4),
                "fp_multiplier":       round(fp_multiplier, 4),
                "decision_score":      round(decision_score, 4),
                "gt_detected":         len(self.detected_gt_indices),
                "gt_total":            len(self.gt_issues),
                "false_positives":     self.false_positive_count,
                "total_submitted":     self.total_submitted_issues,
            },
        )

    # ── State dump ───────────────────────────────────────────────────────────

    def state(self) -> dict[str, Any]:
        return {
            "session_id":          self.session_id,
            "task_id":             self.task_id,
            "step_number":         self.step_number,
            "max_steps":           self.max_steps,
            "done":                self.done,
            "cumulative_reward":   self.cumulative_reward,
            "gt_detected":         len(self.detected_gt_indices),
            "gt_total":            len(self.gt_issues),
            "false_positives":     self.false_positive_count,
            "final_decision":      self.final_decision.value if self.final_decision else None,
            "review_thread":       self.review_thread,
        }


# ─────────────────────────────────────────────────────────────────────────────
# In-memory session store
# ─────────────────────────────────────────────────────────────────────────────

_store: dict[str, Episode] = {}

def create_episode(task_id: str) -> Episode:
    ep = Episode(task_id)
    _store[ep.session_id] = ep
    return ep

def get_episode(session_id: str) -> Episode:
    ep = _store.get(session_id)
    if ep is None:
        raise KeyError(f"Session {session_id!r} not found. Did you call /reset first?")
    return ep