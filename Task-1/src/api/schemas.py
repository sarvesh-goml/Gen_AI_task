"""src/api/schemas.py - request/response models for the FastAPI layer."""

from typing import Any, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    query: str


class ChatResponse(BaseModel):
    reply: str
    citations: Optional[list[str]] = None
    trace: Optional[list[dict[str, Any]]] = None


class SQLQueryRequest(BaseModel):
    question: str


class SQLQueryResponse(BaseModel):
    answer: Optional[str] = None
    sql: Optional[str] = None
    columns: list[str] = []
    rows: list[list[Any]] = []
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    knowledge_base: bool
    groq: bool
    postgres: bool
