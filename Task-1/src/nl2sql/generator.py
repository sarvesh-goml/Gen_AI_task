"""
src/nl2sql/generator.py - turns a natural-language question into a candidate SQL query,
using the same local Ollama wrapper (src/core/llm) every other LLM call in this app uses.
The candidate is untrusted until it passes src/nl2sql/guard.py AND runs under the
read-only DB role - this module's only job is to produce a reasonable first attempt.
"""

import re

from src.core.llm import chat

SQL_SYSTEM_PROMPT = """You are a SQL generator for a PostgreSQL database. Given the schema
below and a question, write EXACTLY ONE read-only SELECT statement that answers it.

Rules:
- Output ONLY the SQL query. No prose, no explanation, no markdown code fences.
- Only a single SELECT (or WITH ... SELECT) statement - never INSERT/UPDATE/DELETE/DROP/
  ALTER, and never more than one statement.
- Use only the tables and columns listed in the schema below - never invent a column.
- If the question cannot be answered from this schema, output exactly: NO_QUERY
"""

FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _extract_sql(raw: str) -> str:
    """Local models often wrap output in markdown fences despite instructions not to -
    strip them defensively rather than trust the instruction alone."""
    match = FENCE_RE.search(raw)
    text = match.group(1) if match else raw
    return text.strip()


def generate_sql(question: str, schema_description: str, previous_sql: str = None, retry_feedback: str = None) -> str:
    messages = [
        {"role": "system", "content": SQL_SYSTEM_PROMPT + "\n" + schema_description},
        {"role": "user", "content": question},
    ]
    if retry_feedback:
        messages.append({"role": "assistant", "content": previous_sql or ""})
        messages.append({
            "role": "user",
            "content": f"That query was rejected: {retry_feedback}. Try again - output "
                       "ONLY a single corrected SELECT statement.",
        })
    raw = chat(messages, temperature=0.0)
    return _extract_sql(raw)
