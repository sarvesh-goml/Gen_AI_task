"""
src/api/main.py - the FastAPI service. Streamlit and the CLI (src/ui/) are both thin HTTP
clients of this one process - neither has any RAG/Agentic/NL2SQL logic of its own, same
separation-of-concerns judgment call the original core/app split made, just with a real
network boundary between "logic" and "presentation" now.

Run from the project root:
    uvicorn src.api.main:app --reload --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config.settings import settings
from src.api.routes import chat, health, sql
from src.core.llm import groq_alive
from src.core.vector_store import collection_ready, get_client, get_embedder
from src.db.database import db_reachable

WORKER_ENV_VARS = ("WEB_CONCURRENCY", "WORKERS", "UVICORN_WORKERS", "GUNICORN_WORKERS")


def _enforce_single_worker():
    """The session store (src/api/session_store.py), the agent's ACTION_LOG
    (src/core/tools.py), and the Qdrant client/embedder singletons (src/core/vector_store.py)
    are all in-process Python state, not backed by a shared store. With more than one
    worker, each gets its own independent copy - a session's memory only "sticks" on
    whichever worker happened to handle that request. Fail fast at startup instead of
    letting that show up later as an unexplained intermittent bug."""
    for var in WORKER_ENV_VARS:
        value = os.environ.get(var)
        if value and value.isdigit() and int(value) > 1:
            raise RuntimeError(
                f"{var}={value} requests multiple workers, but this app keeps session "
                "memory and the Qdrant/embedder singletons as in-process state - it must "
                "run with exactly one worker. Remove the multi-worker setting."
            )
    for i, arg in enumerate(sys.argv):
        if arg == "--workers" and i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
            if int(sys.argv[i + 1]) > 1:
                raise RuntimeError("--workers > 1 is not supported - see _enforce_single_worker's docstring.")


_enforce_single_worker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_embedder()
    get_client()
    print(f"Knowledge base ready: {collection_ready()}")
    print(f"Groq reachable: {groq_alive()} (model: {settings.GROQ_MODEL})")
    print(f"Postgres reachable (readonly role): {db_reachable()}")
    yield


app = FastAPI(title="Spider-Man AI Demo API", version="1.0.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(chat.router, prefix="/api/v1")
app.include_router(sql.router, prefix="/api/v1")
