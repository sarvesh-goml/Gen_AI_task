"""
src/nl2sql/pipeline.py - orchestrates the NL2SQL flow: generate -> guard -> execute
(read-only) -> synthesize a natural-language answer. This is the SQL counterpart to
src/core/rag.answer() - same "retrieve, then generate" shape, different retrieval
backend (a relational DB instead of a vector index).
"""

from sqlalchemy import text

from config.settings import settings
from src.core.llm import chat
from src.db.database import readonly_engine
from src.nl2sql.generator import generate_sql
from src.nl2sql.guard import UnsafeSQLError, validate_select_only
from src.nl2sql.schema_introspection import get_schema_description

ANSWER_SYSTEM_PROMPT = """You are the Spider-Man AI assistant. You were given a question, the exact SQL
query that was run against the structured database, and its results.
Answer the question in one or two sentences, in character (intelligent, witty, responsible, slightly humorous), using ONLY the numbers/rows
shown - never invent a row that isn't there. If the result set is empty, say plainly that
no matching records were found.
"""


def _execute(sql: str):
    with readonly_engine.connect() as conn:
        conn.execute(text(f"SET statement_timeout = {settings.SQL_STATEMENT_TIMEOUT_MS}"))
        result = conn.execute(text(sql))
        columns = list(result.keys())
        rows = [list(row) for row in result.fetchmany(settings.MAX_SQL_ROWS)]
    return columns, rows


def _synthesize_answer(question: str, sql: str, columns, rows) -> str:
    table_preview = f"Columns: {columns}\nRows: {rows}" if rows else "No rows returned."
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nSQL run: {sql}\n\n{table_preview}"},
    ]
    return chat(messages, temperature=0.2)


def answer(question: str) -> dict:
    """Returns {"answer", "sql", "columns", "rows", "error"} - error is None on success."""
    schema = get_schema_description()
    sql = generate_sql(question, schema)

    if sql.strip().upper() == "NO_QUERY":
        return {"answer": None, "sql": None, "columns": [], "rows": [],
                "error": "the question can't be answered from the fleet-ops schema"}

    for attempt in range(2):
        try:
            safe_sql = validate_select_only(sql)
            columns, rows = _execute(safe_sql)
            reply = _synthesize_answer(question, safe_sql, columns, rows)
            return {"answer": reply, "sql": safe_sql, "columns": columns, "rows": rows, "error": None}
        except UnsafeSQLError as e:
            if attempt == 0:
                sql = generate_sql(question, schema, previous_sql=sql, retry_feedback=str(e))
                continue
            return {"answer": None, "sql": sql, "columns": [], "rows": [], "error": str(e)}
        except Exception as e:
            return {"answer": None, "sql": sql, "columns": [], "rows": [], "error": f"query execution failed: {e}"}
