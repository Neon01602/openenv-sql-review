"""
Task bank for the SQL Query Review OpenEnv environment.

Provides:
  - 6 hand-crafted tasks (easy × 2, medium × 2, hard × 2)
  - A procedural task generator that composes novel queries from templates
    so agents cannot simply memorize the 6 fixed tasks.

Each task is a dict with:
  id, difficulty, query, dialect, schema_context, ground_truth_issues,
  correct_decision, instructions, max_steps
"""
from __future__ import annotations
import random
from copy import deepcopy
from typing import Any

from app.models import (
    ColumnInfo, SchemaInfo, ReviewIssue,
    IssueCategory, IssueSeverity, ReviewDecision,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _col(name: str, type_: str, nullable: bool = True) -> ColumnInfo:
    return ColumnInfo(name=name, type=type_, nullable=nullable)

def _table(name, cols, rows=0, indexes=None, constraints=None) -> SchemaInfo:
    return SchemaInfo(
        table_name=name,
        columns=cols,
        row_count_hint=rows,
        has_index_on=indexes or [],
        constraints=constraints or [],
    )


# ─────────────────────────────────────────────────────────────────────────────
# FIXED TASK BANK  (6 tasks)
# ─────────────────────────────────────────────────────────────────────────────

FIXED_TASKS: list[dict[str, Any]] = [

    # ── EASY 1: Wrong JOIN type (nullable FK) ────────────────────────────────
    {
        "id": "easy_correctness",
        "difficulty": "easy",
        "max_steps": 5,
        "dialect": "PostgreSQL",
        "query": """\
-- Fetch all orders with customer info for the monthly report
SELECT
    o.order_id,
    o.created_at,
    o.total_amount,
    c.name      AS customer_name,
    c.email     AS customer_email
FROM orders o
INNER JOIN customers c ON o.customer_id = c.id
WHERE o.created_at >= '2024-01-01'
ORDER BY o.created_at DESC;""",
        "schema_context": [
            _table("orders",
                [_col("order_id","INT PRIMARY KEY",False),
                 _col("customer_id","INT",True),   # nullable — guest checkout
                 _col("created_at","TIMESTAMPTZ",False),
                 _col("total_amount","NUMERIC(10,2)",False),
                 _col("status","VARCHAR(20)",True)],
                rows=2_400_000, indexes=["order_id","created_at"]),
            _table("customers",
                [_col("id","INT PRIMARY KEY",False),
                 _col("name","VARCHAR(120)",False),
                 _col("email","VARCHAR(255)",False)],
                rows=180_000, indexes=["id"]),
        ],
        "ground_truth_issues": [
            ReviewIssue(
                category=IssueCategory.correctness, severity=IssueSeverity.high,
                description="INNER JOIN silently drops orders where customer_id is NULL "
                            "(guest checkouts). Use LEFT JOIN to retain all orders.",
                suggested_fix="Change INNER JOIN to LEFT JOIN on customers.",
            )
        ],
        "correct_decision": ReviewDecision.request_changes,
        "instructions": (
            "Review this SQL query for correctness. "
            "Pay attention to nullable foreign keys and JOIN semantics. "
            "Your final action must include a decision: approve, request_changes, or reject."
        ),
    },

    # ── EASY 2: Implicit type coercion / index defeat ────────────────────────
    {
        "id": "easy_type_coercion",
        "difficulty": "easy",
        "max_steps": 5,
        "dialect": "PostgreSQL",
        "query": """\
-- Look up a user by account code (passed from application layer)
SELECT user_id, username, email
FROM users
WHERE account_code = 10042;  -- application sends an integer""",
        "schema_context": [
            _table("users",
                [_col("user_id","BIGSERIAL PRIMARY KEY",False),
                 _col("account_code","VARCHAR(20)",False),  # stored as text!
                 _col("username","VARCHAR(80)",False),
                 _col("email","VARCHAR(255)",False)],
                rows=5_000_000, indexes=["account_code"]),
        ],
        "ground_truth_issues": [
            ReviewIssue(
                category=IssueCategory.performance, severity=IssueSeverity.high,
                description="Comparing VARCHAR column account_code to an integer literal causes "
                            "implicit type coercion, making the query non-sargable. "
                            "The index on account_code is not used — full sequential scan on 5 M rows.",
                suggested_fix="Quote the literal: WHERE account_code = '10042'",
            ),
            ReviewIssue(
                category=IssueCategory.correctness, severity=IssueSeverity.medium,
                description="Integer-to-text comparison may silently miss leading-zero codes "
                            "such as '010042' that are stored in the column.",
                suggested_fix="Always pass the value as a string from the application layer.",
            ),
        ],
        "correct_decision": ReviewDecision.request_changes,
        "instructions": (
            "Review this SQL query. Consider column types carefully and "
            "whether the WHERE clause predicate is sargable. "
            "Your final action must include a decision."
        ),
    },

    # ── MEDIUM 1: Three performance anti-patterns ────────────────────────────
    {
        "id": "medium_performance",
        "difficulty": "medium",
        "max_steps": 7,
        "dialect": "PostgreSQL",
        "query": """\
-- Dashboard: active users who placed orders in last 30 days (runs every minute)
SELECT
    u.user_id,
    u.username,
    u.email,
    (SELECT COUNT(*) FROM orders o2
     WHERE o2.user_id = u.user_id
       AND o2.created_at > NOW() - INTERVAL '30 days') AS order_count
FROM users u
WHERE LOWER(u.email) = LOWER($1)
  AND u.status = 'active'
ORDER BY u.created_at DESC;""",
        "schema_context": [
            _table("users",
                [_col("user_id","BIGINT PRIMARY KEY",False),
                 _col("username","VARCHAR(80)",False),
                 _col("email","VARCHAR(255)",False),
                 _col("status","VARCHAR(20)",True),
                 _col("created_at","TIMESTAMPTZ",False)],
                rows=50_000_000,
                indexes=["user_id","email"]),
            _table("orders",
                [_col("order_id","BIGINT PRIMARY KEY",False),
                 _col("user_id","BIGINT",False),
                 _col("created_at","TIMESTAMPTZ",False),
                 _col("total","NUMERIC(12,2)",False)],
                rows=200_000_000, indexes=["order_id","user_id","created_at"]),
        ],
        "ground_truth_issues": [
            ReviewIssue(
                category=IssueCategory.performance, severity=IssueSeverity.high,
                description="Correlated subquery in SELECT executes once per outer row — "
                            "classic N+1 pattern. At 50 M users this is catastrophic.",
                suggested_fix="Replace with a LEFT JOIN + COUNT(o.order_id) and GROUP BY.",
            ),
            ReviewIssue(
                category=IssueCategory.performance, severity=IssueSeverity.high,
                description="LOWER(u.email) in WHERE defeats the B-tree index on email, "
                            "forcing a full sequential scan over 50 M rows.",
                suggested_fix="Create a functional index: CREATE INDEX ON users (LOWER(email)); "
                               "or store emails pre-lowercased.",
            ),
            ReviewIssue(
                category=IssueCategory.performance, severity=IssueSeverity.medium,
                description="SELECT u.* implicitly includes all columns — adds unnecessary "
                            "I/O on a 50 M-row table; especially harmful if wide text columns exist.",
                suggested_fix="Select only the columns actually needed by the dashboard.",
            ),
        ],
        "correct_decision": ReviewDecision.request_changes,
        "instructions": (
            "Review this dashboard query for performance issues. "
            "It runs every minute against very large tables. "
            "Identify all anti-patterns, rate their severity, and make a final decision."
        ),
    },

    # ── MEDIUM 2: Aggregation logic bug + missing index ──────────────────────
    {
        "id": "medium_aggregation",
        "difficulty": "medium",
        "max_steps": 7,
        "dialect": "PostgreSQL",
        "query": """\
-- Weekly revenue summary by product category
SELECT
    p.category,
    SUM(oi.quantity * oi.unit_price)  AS revenue,
    COUNT(DISTINCT o.order_id)        AS order_count,
    AVG(oi.quantity * oi.unit_price)  AS avg_order_value
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN products p    ON oi.product_id = p.product_id
WHERE o.created_at BETWEEN '2024-01-01' AND '2024-01-07'
GROUP BY p.category
HAVING COUNT(oi.item_id) > 0
ORDER BY revenue DESC;""",
        "schema_context": [
            _table("orders",
                [_col("order_id","BIGINT PRIMARY KEY",False),
                 _col("created_at","TIMESTAMPTZ",False),
                 _col("status","VARCHAR(20)",False)],
                rows=8_000_000, indexes=["order_id"]),          # NO index on created_at!
            _table("order_items",
                [_col("item_id","BIGINT PRIMARY KEY",False),
                 _col("order_id","BIGINT",False),
                 _col("product_id","BIGINT",False),
                 _col("quantity","INT",False),
                 _col("unit_price","NUMERIC(10,2)",False)],
                rows=40_000_000, indexes=["item_id","order_id","product_id"]),
            _table("products",
                [_col("product_id","BIGINT PRIMARY KEY",False),
                 _col("name","VARCHAR(200)",False),
                 _col("category","VARCHAR(80)",False)],
                rows=120_000, indexes=["product_id"]),
        ],
        "ground_truth_issues": [
            ReviewIssue(
                category=IssueCategory.correctness, severity=IssueSeverity.high,
                description="HAVING COUNT(oi.item_id) > 0 is a no-op after an INNER JOIN — "
                            "every group already has at least one item. If the intent was to "
                            "exclude categories with zero revenue, this filter does nothing.",
                suggested_fix="Remove the HAVING clause or rethink with a LEFT JOIN if zero-revenue "
                               "categories should appear.",
            ),
            ReviewIssue(
                category=IssueCategory.correctness, severity=IssueSeverity.medium,
                description="AVG(oi.quantity * oi.unit_price) computes the average per line item, "
                            "not the average order value. For a true average order value use "
                            "SUM(...) / COUNT(DISTINCT o.order_id).",
                suggested_fix="Replace AVG(...) with SUM(oi.quantity * oi.unit_price) / NULLIF(COUNT(DISTINCT o.order_id),0).",
            ),
            ReviewIssue(
                category=IssueCategory.performance, severity=IssueSeverity.high,
                description="No index on orders.created_at — the BETWEEN predicate triggers "
                            "a full sequential scan on 8 M rows every time this report runs.",
                suggested_fix="CREATE INDEX ON orders (created_at);",
            ),
            ReviewIssue(
                category=IssueCategory.correctness, severity=IssueSeverity.low,
                description="BETWEEN is inclusive on both ends; if created_at is a TIMESTAMPTZ, "
                            "orders at exactly '2024-01-07 00:00:00' are included but orders "
                            "placed on Jan 7th after midnight are excluded. "
                            "Use >= and < '2024-01-08' for a full day range.",
                suggested_fix="Replace BETWEEN with: created_at >= '2024-01-01' AND created_at < '2024-01-08'",
            ),
        ],
        "correct_decision": ReviewDecision.request_changes,
        "instructions": (
            "Review this weekly revenue report query. Focus on correctness of aggregation "
            "logic and potential performance issues. Your final decision must reflect "
            "the combined severity of all issues found."
        ),
    },

    # ── HARD 1: Security + logic + performance (multi-bug) ──────────────────
    {
        "id": "hard_security",
        "difficulty": "hard",
        "max_steps": 10,
        "dialect": "PostgreSQL",
        "query": """\
-- Public product search API endpoint
CREATE OR REPLACE FUNCTION search_products(
    search_term TEXT,
    sort_by     TEXT DEFAULT 'price',
    min_rating  FLOAT DEFAULT 0.0
)
RETURNS TABLE(product_id BIGINT, name TEXT, price NUMERIC, password_hash TEXT) AS $$
DECLARE
    sql TEXT;
BEGIN
    sql := format(
        'SELECT p.product_id, p.name, p.price, u.password_hash
         FROM products p
         JOIN users u ON p.seller_id = u.user_id
         LEFT JOIN reviews r ON r.product_id = p.product_id
         WHERE p.name ILIKE ''%%%s%%''
           AND p.is_active = true
         GROUP BY p.product_id, p.name, p.price, u.password_hash
         HAVING COUNT(r.id) > 0
         ORDER BY %s ASC',
        search_term,   -- ← injected directly
        sort_by        -- ← injected directly
    );
    RETURN QUERY EXECUTE sql;
END;
$$ LANGUAGE plpgsql;""",
        "schema_context": [
            _table("products",
                [_col("product_id","BIGINT PRIMARY KEY",False),
                 _col("seller_id","BIGINT",False),
                 _col("name","TEXT",False),
                 _col("price","NUMERIC(12,2)",False),
                 _col("is_active","BOOLEAN",False)],
                rows=500_000, indexes=["product_id","seller_id","name"]),
            _table("users",
                [_col("user_id","BIGINT PRIMARY KEY",False),
                 _col("email","VARCHAR(255)",False),
                 _col("password_hash","TEXT",False),
                 _col("role","VARCHAR(20)",False)],
                rows=2_000_000, indexes=["user_id","email"]),
            _table("reviews",
                [_col("id","BIGINT PRIMARY KEY",False),
                 _col("product_id","BIGINT",False),
                 _col("user_id","BIGINT",False),
                 _col("rating","FLOAT",False),
                 _col("created_at","TIMESTAMPTZ",False)],
                rows=10_000_000, indexes=["id","product_id"]),
        ],
        "ground_truth_issues": [
            ReviewIssue(
                category=IssueCategory.security, severity=IssueSeverity.critical,
                description="search_term is interpolated directly into the dynamic SQL string via "
                            "format('%s', search_term) — classic SQL injection. An attacker can "
                            "escape the ILIKE and execute arbitrary SQL as the database superuser.",
                suggested_fix="Use parameterised EXECUTE with USING: "
                               "EXECUTE sql USING search_term; and replace %s with $1 in the format string.",
            ),
            ReviewIssue(
                category=IssueCategory.security, severity=IssueSeverity.critical,
                description="password_hash is selected and returned in the function's result set. "
                            "The public API will expose all seller password hashes to any caller.",
                suggested_fix="Remove password_hash from SELECT and from the RETURNS TABLE signature entirely.",
            ),
            ReviewIssue(
                category=IssueCategory.correctness, severity=IssueSeverity.high,
                description="HAVING COUNT(r.id) > 0 combined with LEFT JOIN excludes products "
                            "that have no reviews. Likely the intent was to show all active products; "
                            "the HAVING clause silently hides new/unreviewed products.",
                suggested_fix="Remove the HAVING clause or change to HAVING COUNT(r.id) >= 0 "
                               "if filtering by min_rating is desired.",
            ),
            ReviewIssue(
                category=IssueCategory.performance, severity=IssueSeverity.medium,
                description="sort_by is user-controlled and interpolated directly — allows the "
                            "caller to sort on any expression, including unindexed columns or "
                            "expressions that force a filesort on 500 K+ rows. Also enables "
                            "second-order injection.",
                suggested_fix="Whitelist allowed sort columns: "
                               "IF sort_by NOT IN ('price','name','rating') THEN RAISE EXCEPTION ...; END IF;",
            ),
        ],
        "correct_decision": ReviewDecision.reject,
        "instructions": (
            "Review this PostgreSQL stored function used by a public-facing API. "
            "This code handles external user input. Look for security vulnerabilities, "
            "correctness bugs, and performance issues. "
            "Both critical security issues must be caught for a high score. "
            "Your final decision must reflect the severity of all issues found."
        ),
    },

    # ── HARD 2: Schema migration script with data-loss risk ──────────────────
    {
        "id": "hard_migration",
        "difficulty": "hard",
        "max_steps": 10,
        "dialect": "PostgreSQL",
        "query": """\
-- Migration: split full_name into first_name / last_name + add audit log
BEGIN;

ALTER TABLE users ADD COLUMN first_name VARCHAR(80);
ALTER TABLE users ADD COLUMN last_name  VARCHAR(80);

UPDATE users
SET
    first_name = SPLIT_PART(full_name, ' ', 1),
    last_name  = SPLIT_PART(full_name, ' ', 2);

ALTER TABLE users DROP COLUMN full_name;

ALTER TABLE users
    ALTER COLUMN first_name SET NOT NULL,
    ALTER COLUMN last_name  SET NOT NULL;

CREATE TABLE audit_log (
    id         SERIAL PRIMARY KEY,
    user_id    INT,
    action     TEXT,
    changed_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO audit_log (user_id, action)
SELECT user_id, 'name_split_migration'
FROM users;

COMMIT;""",
        "schema_context": [
            _table("users",
                [_col("user_id","SERIAL PRIMARY KEY",False),
                 _col("full_name","VARCHAR(200)",True),    # nullable!
                 _col("email","VARCHAR(255)",False),
                 _col("created_at","TIMESTAMPTZ",False)],
                rows=12_000_000, indexes=["user_id","email"],
                constraints=["FK: orders.user_id → users.user_id"]),
        ],
        "ground_truth_issues": [
            ReviewIssue(
                category=IssueCategory.correctness, severity=IssueSeverity.critical,
                description="SPLIT_PART(full_name, ' ', 2) returns empty string '' for single-name "
                            "users (e.g. 'Madonna', 'Cher') or users with NULL full_name. "
                            "The subsequent NOT NULL constraint will then fail or silently "
                            "store empty strings as last names, corrupting 12 M rows.",
                suggested_fix="Add NULLIF handling and a pre-migration check: "
                               "SELECT COUNT(*) FROM users WHERE full_name IS NULL OR full_name NOT LIKE '% %';",
            ),
            ReviewIssue(
                category=IssueCategory.correctness, severity=IssueSeverity.high,
                description="DROP COLUMN full_name is irreversible inside this transaction. "
                            "If any step after it fails (e.g. NOT NULL constraint), "
                            "the rollback restores full_name but the data split is lost. "
                            "More critically, there is no backup strategy before the destructive DROP.",
                suggested_fix="Keep full_name as a nullable column (renamed to full_name_legacy) "
                               "for at least one release cycle before dropping.",
            ),
            ReviewIssue(
                category=IssueCategory.performance, severity=IssueSeverity.high,
                description="ALTER TABLE ... SET NOT NULL on a 12 M-row table requires a full "
                            "table rewrite and takes an ACCESS EXCLUSIVE lock. This will block "
                            "all reads and writes on users for several minutes in production.",
                suggested_fix="Use NOT VALID constraints + VALIDATE CONSTRAINT in a separate "
                               "transaction, or add a CHECK constraint with NOT VALID first.",
            ),
            ReviewIssue(
                category=IssueCategory.correctness, severity=IssueSeverity.medium,
                description="audit_log.user_id is INT but users.user_id is SERIAL (INT). "
                            "No foreign key constraint is defined — orphaned audit rows are "
                            "possible if users are later deleted.",
                suggested_fix="Add: FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL",
            ),
        ],
        "correct_decision": ReviewDecision.reject,
        "instructions": (
            "Review this production database migration script. It runs against a 12 M-row "
            "users table with live traffic. Look for data-loss risks, locking issues, "
            "correctness bugs, and missing safeguards. "
            "Be especially careful about irreversible operations and constraint violations."
        ),
    },
]

# Build lookup
TASK_MAP: dict[str, dict] = {t["id"]: t for t in FIXED_TASKS}


# ─────────────────────────────────────────────────────────────────────────────
# PROCEDURAL TASK GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
# Generates novel queries by composing bug templates with randomised schemas
# so agents cannot simply memorise the 6 fixed tasks.

_BUG_TEMPLATES = [
    # (difficulty, category, severity, description_template, fix_template, sql_snippet, decision)
    ("easy", "correctness", "high",
     "INNER JOIN on nullable FK {col} silently drops rows where {col} IS NULL.",
     "Change INNER JOIN to LEFT JOIN.",
     "INNER JOIN {ref_table} r ON t.{col} = r.id",
     "request_changes"),
    ("easy", "performance", "high",
     "LOWER({col}) in WHERE defeats the index on {col}, causing a full seq scan.",
     "Create a functional index on LOWER({col}) or store the value pre-lowercased.",
     "WHERE LOWER(t.{col}) = LOWER($1)",
     "request_changes"),
    ("medium", "security", "critical",
     "{col} is interpolated directly into a dynamic SQL string — SQL injection risk.",
     "Use parameterised queries (EXECUTE ... USING) instead of string interpolation.",
     "EXECUTE 'SELECT ... WHERE x = ' || {col}",
     "reject"),
    ("medium", "correctness", "high",
     "HAVING COUNT(...) > 0 after INNER JOIN is a no-op and filters unintentionally.",
     "Remove the HAVING clause or switch to LEFT JOIN if zero-count rows are needed.",
     "HAVING COUNT(sub.id) > 0",
     "request_changes"),
    ("hard", "security", "critical",
     "Sensitive column {col} is exposed in the SELECT — leaks private data to callers.",
     "Remove {col} from the SELECT list and from the API response schema.",
     "SELECT t.{col}, ...",
     "reject"),
    ("hard", "performance", "high",
     "No index on {col} used in WHERE/ORDER BY — causes full seq scan on large table.",
     "CREATE INDEX ON {table}({col});",
     "ORDER BY t.{col} DESC",
     "request_changes"),
]

def generate_task(seed: int | None = None) -> dict[str, Any]:
    """Return a novel, fully-formed task dict generated procedurally."""
    rng = random.Random(seed)

    # Pick 1–3 bugs
    n_bugs = rng.randint(1, 3)
    chosen = rng.sample(_BUG_TEMPLATES, min(n_bugs, len(_BUG_TEMPLATES)))

    table_names = ["events", "transactions", "sessions", "accounts", "shipments",
                   "subscriptions", "payments", "employees", "tickets", "invoices"]
    col_names   = ["user_id", "account_id", "email", "status", "token",
                   "secret_key", "ssn", "phone", "ip_address", "api_key"]

    table = rng.choice(table_names)
    col   = rng.choice(col_names)

    difficulty_rank = {"easy": 0, "medium": 1, "hard": 2}
    max_diff = max(chosen, key=lambda b: difficulty_rank[b[0]])[0]
    max_steps = {"easy": 5, "medium": 7, "hard": 10}[max_diff]

    # Determine worst decision needed
    decisions = [b[6] for b in chosen]
    decision = "reject" if "reject" in decisions else "request_changes"

    issues = []
    for bug in chosen:
        _, cat, sev, desc_t, fix_t, _, _ = bug
        issues.append(ReviewIssue(
            category=IssueCategory(cat),
            severity=IssueSeverity(sev),
            description=desc_t.format(col=col, table=table, ref_table=table+"_refs"),
            suggested_fix=fix_t.format(col=col, table=table),
        ))

    # Build a simple schema
    schema = [
        _table(table,
            [_col("id","BIGINT PRIMARY KEY",False),
             _col(col,"VARCHAR(255)",True),
             _col("created_at","TIMESTAMPTZ",False),
             _col("status","VARCHAR(20)",True)],
            rows=rng.choice([100_000, 1_000_000, 10_000_000, 50_000_000]),
            indexes=["id"]),
    ]

    # Generate a query string that contains the bug snippets
    snippet_lines = "\n".join(b[5].format(col=col, table=table, ref_table=table+"_refs") for b in chosen)
    query = f"""\
-- Auto-generated query (seed={seed})
SELECT t.*
FROM {table} t
{snippet_lines}
WHERE t.created_at > NOW() - INTERVAL '7 days';"""

    task_id = f"generated_{max_diff}_{abs(seed or rng.randint(0,99999)):05d}"

    return {
        "id":               task_id,
        "difficulty":       max_diff,
        "max_steps":        max_steps,
        "dialect":          "PostgreSQL",
        "query":            query,
        "schema_context":   schema,
        "ground_truth_issues": issues,
        "correct_decision": ReviewDecision(decision),
        "instructions": (
            f"Review this SQL query against the {table} table. "
            "Identify all correctness, performance, and security issues. "
            "Rate severity accurately and provide a final decision."
        ),
    }


def get_task(task_id: str) -> dict[str, Any]:
    """Return a task by id.  If not in TASK_MAP, try to parse as generated_*."""
    if task_id in TASK_MAP:
        return deepcopy(TASK_MAP[task_id])

    # generated_<difficulty>_<seed>
    parts = task_id.split("_")
    if len(parts) == 3 and parts[0] == "generated":
        try:
            seed = int(parts[2])
            return generate_task(seed)
        except ValueError:
            pass

    raise KeyError(f"Unknown task_id: {task_id!r}. "
                   f"Valid fixed IDs: {list(TASK_MAP)} "
                   f"or use 'generated_<easy|medium|hard>_<seed>'.")


def list_tasks() -> list[dict[str, Any]]:
    """Return the action schema + metadata for all fixed tasks."""
    result = []
    for t in FIXED_TASKS:
        result.append({
            "id":           t["id"],
            "difficulty":   t["difficulty"],
            "max_steps":    t["max_steps"],
            "dialect":      t["dialect"],
            "n_ground_truth_issues": len(t["ground_truth_issues"]),
            "correct_decision":      t["correct_decision"],
            "instructions":          t["instructions"],
        })
    return result


ACTION_SCHEMA = {
    "session_id": "string — must match the current episode session_id",
    "issues": [
        {
            "category":     "enum: correctness | performance | security | style",
            "severity":     "enum: critical | high | medium | low",
            "line_hint":    "int (optional) — 1-indexed line number in the query",
            "description":  "string — explanation of the issue",
            "suggested_fix":"string (optional) — how to fix it",
        }
    ],
    "comment":  "string (optional) — overall review comment",
    "decision": "enum (optional): approve | request_changes | reject — submitting ends the episode",
}