"""
src/nl2sql/guard.py - text-level safety checks on LLM-generated SQL.

This is DEFENSE IN DEPTH, not the actual safety boundary - it's a fast, cheap first line
of defense that rejects the obvious cases before a query ever reaches the database. The
real boundary is that src/db/database.readonly_engine connects as a PostgreSQL role
(jarvis_readonly, created by scripts/setup_db.py) with SELECT-only grants on the 4 fleet
tables - so even a generated statement this guard fails to catch cannot write or drop
anything, because the role it runs as has no privilege to.
"""

import re

BANNED_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "GRANT", "REVOKE", "ATTACH", "COPY", "VACUUM", "REPLACE", "EXEC", "CALL",
)
BANNED_RE = re.compile(r"\b(" + "|".join(BANNED_KEYWORDS) + r")\b", re.IGNORECASE)


class UnsafeSQLError(ValueError):
    pass


def validate_select_only(sql: str) -> str:
    """Returns the cleaned SQL if it passes, else raises UnsafeSQLError with a reason
    specific enough that the caller can feed it back to the LLM for a retry."""
    cleaned = sql.strip()
    if not cleaned:
        raise UnsafeSQLError("empty query")

    # Allow exactly one trailing semicolon; anything after/another one means multiple
    # statements were stacked, which a single execute() call must never run.
    body = cleaned[:-1].strip() if cleaned.endswith(";") else cleaned
    if ";" in body:
        raise UnsafeSQLError("multiple statements are not allowed (found an embedded ';')")

    if not re.match(r"^\s*(SELECT|WITH)\b", body, re.IGNORECASE):
        raise UnsafeSQLError("only a single SELECT (or WITH ... SELECT) statement is allowed")

    banned = BANNED_RE.search(body)
    if banned:
        raise UnsafeSQLError(f"'{banned.group(1).upper()}' is not allowed in a read-only query")

    return body
