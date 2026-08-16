"""
src/api/routes/sql.py - POST /api/v1/sql/query, a direct entry point into the NL2SQL
pipeline (src/nl2sql/pipeline.answer). Exists mainly so the fleet DB can be exercised on
its own with a plain curl call, the same way `lookup_knowledge_base` can be tested via
RAG mode without going through the agent - see README Sec 8 for example curl commands.
"""

from fastapi import APIRouter

from src.api.schemas import SQLQueryRequest, SQLQueryResponse
from src.nl2sql.pipeline import answer as nl2sql_answer

router = APIRouter()


@router.post("/sql/query", response_model=SQLQueryResponse)
def sql_query(req: SQLQueryRequest):
    result = nl2sql_answer(req.question)
    return SQLQueryResponse(**result)
