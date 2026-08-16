"""
src/core/rag.py - the RAG pipeline: Retrieval & Context Injection, then Generation & Formatting.
"""

from src.core.vector_store import retrieve
from src.core.llm import chat
from config.settings import settings

SPIDEY_SYSTEM_PROMPT = settings.SPIDEY_SYSTEM_PROMPT
TOP_K = settings.TOP_K


def format_context(chunks):
    """Context injection: format retrieved chunks with clear source labels."""
    if not chunks:
        return "No relevant information found in Spider-Man's knowledge base."
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] (source: {c['doc_type']}) {c['text']}")
    return "\n".join(lines)


def answer(query, top_k=TOP_K, history=None):
    """Retrieval & Context Injection -> Generation & Formatting."""
    chunks = retrieve(query, top_k=top_k)
    context = format_context(chunks)

    messages = [{"role": "system", "content": SPIDEY_SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nAnswer using ONLY the context above.",
    })

    reply = chat(messages)
    return reply, chunks
