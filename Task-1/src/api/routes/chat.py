"""
src/api/routes/chat.py - POST /api/v1/chat/rag and /api/v1/chat/agent, plus session
clearing. This is the only place RAG vs. Agentic mode is chosen - the UI/CLI are thin
HTTP clients that never touch src/core directly (see src/ui/).
"""

from fastapi import APIRouter

from src.api.schemas import ChatRequest, ChatResponse
from src.api.session_store import clear_session, get_session
from src.core.agent import run_agent
from src.core.rag import answer as rag_answer

router = APIRouter()


@router.post("/chat/rag", response_model=ChatResponse)
def chat_rag(req: ChatRequest):
    mem = get_session(f"{req.session_id}:rag")
    reply, chunks = rag_answer(req.query, history=mem.as_messages())
    mem.add("user", req.query)
    mem.add("assistant", reply)
    citations = sorted({c["doc_type"] for c in chunks}) if chunks else None
    return ChatResponse(reply=reply, citations=citations)


@router.post("/chat/agent", response_model=ChatResponse)
def chat_agent(req: ChatRequest):
    mem = get_session(f"{req.session_id}:agent")
    reply, trace = run_agent(req.query, mem, verbose=False)
    mem.add("user", req.query)
    mem.add("assistant", reply)
    return ChatResponse(reply=reply, trace=trace)


@router.delete("/chat/session/{session_id}")
def clear(session_id: str, mode: str):
    clear_session(f"{session_id}:{mode}")
    return {"cleared": True}
