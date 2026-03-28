"""
Heuristic baseline for SQL Query Review OpenEnv.

Uses regex/pattern matching against the SQL text and schema.
No API key required — fully deterministic and reproducible.

Runs against all 6 fixed tasks and returns scores.
"""
from __future__ import annotations
import re
from typing import Any

from app.models import ReviewIssue, IssueCategory, IssueSeverity, ReviewDecision
from app.session import create_episode
from app.tasks import FIXED_TASKS


# ─────────────────────────────────────────────────────────────────────────────
# Pattern bank
# ─────────────────────────────────────────────────────────────────────────────

def _detect_issues(query: str, schema_context: list) -> list[ReviewIssue]:
    """Apply heuristic rules to a SQL query and return detected issues."""
    q_upper = query.upper()
    issues: list[ReviewIssue] = []

    # Rule 1: INNER JOIN on a nullable FK
    if re.search(r"INNER\s+JOIN", q_upper):
        # Check schema for nullable columns
        for tbl in schema_context:
            for col in tbl.columns:
                if col.nullable and ("id" in col.name.lower() or "fk" in col.name.lower()):
                    issues.append(ReviewIssue(
                        category=IssueCategory.correctness,
                        severity=IssueSeverity.high,
                        description=(
                            f"INNER JOIN may silently drop rows where {col.name} is NULL. "
                            f"If {col.name} is a nullable foreign key, use LEFT JOIN instead."
                        ),
                        suggested_fix="Change INNER JOIN to LEFT JOIN.",
                    ))
                    break

    # Rule 2: LOWER() on indexed column
    if re.search(r"LOWER\s*\(", q_upper):
        for tbl in schema_context:
            for idx_col in tbl.has_index_on:
                if re.search(rf"LOWER\s*\(\s*\w+\.?{re.escape(idx_col)}\s*\)", query, re.I):
                    issues.append(ReviewIssue(
                        category=IssueCategory.performance,
                        severity=IssueSeverity.high,
                        description=(
                            f"LOWER({idx_col}) in WHERE defeats the B-tree index on {idx_col}, "
                            f"causing a full sequential scan."
                        ),
                        suggested_fix=f"Create a functional index: CREATE INDEX ON {tbl.table_name} (LOWER({idx_col}));",
                    ))

    # Rule 3: Correlated subquery
    if re.search(r"SELECT\s+.*\(\s*SELECT\s+COUNT", q_upper, re.DOTALL):
        issues.append(ReviewIssue(
            category=IssueCategory.performance,
            severity=IssueSeverity.high,
            description=(
                "Correlated subquery in SELECT/WHERE executes once per outer row — "
                "N+1 pattern that is catastrophic at scale."
            ),
            suggested_fix="Replace with a JOIN + GROUP BY or a window function.",
        ))

    # Rule 4: SELECT * on large table
    if re.search(r"SELECT\s+\*", q_upper) or re.search(r"SELECT\s+\w+\.\*", q_upper):
        for tbl in schema_context:
            if tbl.row_count_hint > 1_000_000:
                issues.append(ReviewIssue(
                    category=IssueCategory.performance,
                    severity=IssueSeverity.medium,
                    description=(
                        f"SELECT * on {tbl.table_name} ({tbl.row_count_hint:,} rows) "
                        f"pulls all columns unnecessarily, increasing I/O and memory usage."
                    ),
                    suggested_fix="Select only the columns required.",
                ))
                break

    # Rule 5: SQL injection via string concatenation / format
    if re.search(r"(EXECUTE\s+format|EXECUTE\s+.*\|\||format\s*\(.*%s)", query, re.I | re.DOTALL):
        issues.append(ReviewIssue(
            category=IssueCategory.security,
            severity=IssueSeverity.critical,
            description=(
                "Dynamic SQL built with format('%s', user_input) or string concatenation "
                "is vulnerable to SQL injection. User-supplied values are not sanitised."
            ),
            suggested_fix="Use EXECUTE ... USING with positional parameters ($1, $2, ...).",
        ))

    # Rule 6: Sensitive columns in SELECT (password, secret, token, ssn, hash)
    sensitive_re = re.compile(r"\b(password_hash|password|secret_?key|ssn|api_?key|token)\b", re.I)
    matches = sensitive_re.findall(query)
    if matches:
        col_name = matches[0]
        issues.append(ReviewIssue(
            category=IssueCategory.security,
            severity=IssueSeverity.critical,
            description=(
                f"Sensitive column '{col_name}' is selected and returned by this query. "
                f"This leaks private data to API callers."
            ),
            suggested_fix=f"Remove '{col_name}' from the SELECT list.",
        ))

    # Rule 7: HAVING COUNT > 0 after JOIN (no-op / logic bug)
    if re.search(r"HAVING\s+COUNT\s*\(.*\)\s*>\s*0", q_upper):
        issues.append(ReviewIssue(
            category=IssueCategory.correctness,
            severity=IssueSeverity.high,
            description=(
                "HAVING COUNT(...) > 0 after a JOIN is typically a no-op because "
                "every group already has at least one row. It may also unintentionally "
                "exclude groups from a LEFT JOIN."
            ),
            suggested_fix=(
                "Remove the HAVING clause if filtering is not needed, "
                "or use a LEFT JOIN with HAVING COUNT(...) = 0 to find empty groups."
            ),
        ))

    # Rule 8: Missing index on ORDER BY / WHERE column
    if re.search(r"ORDER\s+BY", q_upper) or re.search(r"WHERE.*created_at", q_upper, re.I):
        for tbl in schema_context:
            if tbl.row_count_hint > 1_000_000:
                order_by_cols = re.findall(r"ORDER\s+BY\s+([\w.]+)", query, re.I)
                where_cols = re.findall(r"WHERE.*?([\w.]+)\s*(?:BETWEEN|>=|<=|=|>|<)", query, re.I)
                for c in order_by_cols + where_cols:
                    col_clean = c.split(".")[-1]
                    if col_clean not in tbl.has_index_on:
                        issues.append(ReviewIssue(
                            category=IssueCategory.performance,
                            severity=IssueSeverity.high,
                            description=(
                                f"Column '{col_clean}' used in ORDER BY / WHERE is not indexed "
                                f"on {tbl.table_name} ({tbl.row_count_hint:,} rows). "
                                f"This causes a full sequential scan."
                            ),
                            suggested_fix=f"CREATE INDEX ON {tbl.table_name} ({col_clean});",
                        ))

    # Rule 9: DROP COLUMN without backup warning
    if re.search(r"DROP\s+COLUMN", q_upper):
        issues.append(ReviewIssue(
            category=IssueCategory.correctness,
            severity=IssueSeverity.critical,
            description=(
                "DROP COLUMN is irreversible. If any prior step in this migration fails "
                "or if data was not fully migrated, the column's data is permanently lost."
            ),
            suggested_fix=(
                "Rename to a _legacy column first and drop in a follow-up migration "
                "after verifying correctness."
            ),
        ))

    # Rule 10: NOT NULL on large table (locking risk)
    if re.search(r"SET\s+NOT\s+NULL", q_upper):
        for tbl in schema_context:
            if tbl.row_count_hint > 1_000_000:
                issues.append(ReviewIssue(
                    category=IssueCategory.performance,
                    severity=IssueSeverity.high,
                    description=(
                        f"ALTER TABLE SET NOT NULL on {tbl.table_name} "
                        f"({tbl.row_count_hint:,} rows) requires a full table rewrite "
                        f"and takes an ACCESS EXCLUSIVE lock, blocking all reads/writes."
                    ),
                    suggested_fix=(
                        "Use NOT VALID + VALIDATE CONSTRAINT in a separate transaction, "
                        "or add a CHECK constraint first."
                    ),
                ))

    # Deduplicate by (category, first 40 chars of description)
    seen: set[tuple] = set()
    deduped: list[ReviewIssue] = []
    for iss in issues:
        key = (iss.category, iss.description[:40])
        if key not in seen:
            seen.add(key)
            deduped.append(iss)

    return deduped


def _pick_decision(issues: list[ReviewIssue]) -> ReviewDecision:
    """Pick a decision based on detected issue severities."""
    if not issues:
        return ReviewDecision.approve
    severities = {i.severity for i in issues}
    if IssueSeverity.critical in severities:
        return ReviewDecision.reject
    if IssueSeverity.high in severities:
        return ReviewDecision.request_changes
    return ReviewDecision.request_changes


# ─────────────────────────────────────────────────────────────────────────────
# Run baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_heuristic_baseline() -> list[dict[str, Any]]:
    """Run the heuristic agent against all 6 fixed tasks. Returns list of results."""
    results = []
    for task_def in FIXED_TASKS:
        ep = create_episode(task_def["id"])
        schema = ep.task["schema_context"]
        query  = ep.task["query"]

        detected = _detect_issues(query, schema)
        decision = _pick_decision(detected)

        from app.models import Action
        action = Action(
            session_id = ep.session_id,
            issues     = detected,
            comment    = f"Heuristic baseline detected {len(detected)} issue(s).",
            decision   = decision,
        )
        _, reward, done, _ = ep.step(action)
        grader = ep.grade()

        results.append({
            "task_id":    task_def["id"],
            "difficulty": task_def["difficulty"],
            "score":      grader.score,
            "breakdown":  grader.breakdown,
            "n_detected": len(detected),
            "decision":   decision.value,
            "reward":     reward.value,
        })

    return results