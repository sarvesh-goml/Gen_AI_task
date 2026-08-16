"""
tests/unit/test_nl2sql_guard.py - the guard is defense-in-depth (the real boundary is the
read-only Postgres role, see src/db/database.py's docstring), but it should still reject
the obvious cases fast, without ever touching the database.
"""

import pytest

from src.nl2sql.guard import UnsafeSQLError, validate_select_only


def test_accepts_simple_select():
    assert validate_select_only("SELECT * FROM suits") == "SELECT * FROM suits"


def test_accepts_select_with_trailing_semicolon():
    assert validate_select_only("SELECT * FROM suits;") == "SELECT * FROM suits"


def test_accepts_cte_with_select():
    sql = "WITH t AS (SELECT 1) SELECT * FROM t"
    assert validate_select_only(sql) == sql


@pytest.mark.parametrize("sql", [
    "DROP TABLE suits",
    "DELETE FROM suits",
    "UPDATE suits SET status = 'combat_ready'",
    "INSERT INTO suits (mark_name) VALUES ('Mark 99')",
    "ALTER TABLE suits ADD COLUMN hacked text",
    "GRANT ALL ON suits TO public",
])
def test_rejects_non_select_statements(sql):
    with pytest.raises(UnsafeSQLError):
        validate_select_only(sql)


def test_rejects_stacked_statements():
    with pytest.raises(UnsafeSQLError):
        validate_select_only("SELECT * FROM suits; DROP TABLE suits;")


def test_rejects_empty_query():
    with pytest.raises(UnsafeSQLError):
        validate_select_only("   ")


def test_does_not_false_positive_on_column_names_containing_banned_substrings():
    # "updated_at" contains "update" but is not the UPDATE keyword - word-boundary
    # matching must not reject it.
    sql = "SELECT updated_at FROM suits"
    assert validate_select_only(sql) == sql
