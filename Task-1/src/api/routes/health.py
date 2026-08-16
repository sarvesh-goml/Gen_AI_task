"""src/api/routes/health.py - GET /health, used by the UI's sidebar status panel."""

from fastapi import APIRouter

from src.api.schemas import HealthResponse
from src.core.llm import groq_alive
from src.core.vector_store import collection_ready
from src.db.database import db_reachable

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        knowledge_base=collection_ready(),
        groq=groq_alive(),
        postgres=db_reachable(),
    )
