"""
config/settings.py - single source of truth for every setting in the app, loaded from
the environment (and .env, via pydantic-settings) instead of hardcoded constants.

Import the shared instance: `from config.settings import settings`.
"""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict

# jarvis_demo/ project root (two levels up from this file: config/settings.py -> config/ -> root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=os.path.join(BASE_DIR, ".env"), extra="ignore")

    # ---- Embeddings ----
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # open-source, ~80MB, CPU-friendly

    # ---- Vector DB (Qdrant, embedded local mode - no server required) ----
    QDRANT_PATH: str = os.path.join(BASE_DIR, "qdrant_data")
    COLLECTION_NAME: str = "jarvis_kb"

    # ---- LLM (Groq) ----
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # ---- Documents ----
    DOCUMENTS_DIR: str = os.path.join(BASE_DIR, "data", "documents")

    # ---- Retrieval ----
    TOP_K: int = 4  # how many chunks to retrieve per query

    # ---- Structured fleet-ops DB (PostgreSQL, via docker-compose) ----
    # Admin/owner connection - used only by scripts/setup_db.py and scripts/seed_db.py.
    # Host port is 5433, not the default 5432 - see docker-compose.yml for why.
    DATABASE_URL: str = "postgresql+psycopg://jarvis:jarvis@localhost:5433/jarvis_fleet"
    # Least-privilege connection actually used to RUN generated SQL (Sec 2's real safety
    # boundary, not just the regex guard). If left unset, derived from DATABASE_URL by
    # swapping in READONLY_DB_USER/READONLY_DB_PASSWORD against the same host/db.
    READONLY_DATABASE_URL: str = ""
    READONLY_DB_USER: str = "jarvis_readonly"
    READONLY_DB_PASSWORD: str = "jarvis_readonly_pw"
    SQL_STATEMENT_TIMEOUT_MS: int = 5000
    MAX_SQL_ROWS: int = 200

    # ---- API / UI wiring ----
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_BASE_URL: str = "http://localhost:8000"

    # ---- Spider-Man AI persona (system prompt) ----
    SPIDEY_SYSTEM_PROMPT: str = """You are the Spider-Man AI assistant.
Speak with intelligence, responsibility, dry wit, and a slightly humorous/friendly tone (like Peter Parker).
Always answer ONLY using the CONTEXT provided below. If the context does not contain the answer, say so plainly instead of guessing - a wrong answer delivered confidently is worse than an honest "I don't have that on file."

Keep factual/technical answers precise and cite which knowledge type answered (e.g. suit diagnostics, combat strategy, mission debriefs) in one short clause. Let personality flavor the delivery, never the facts.
"""

    def readonly_database_url(self) -> str:
        if self.READONLY_DATABASE_URL:
            return self.READONLY_DATABASE_URL
        # Swap the admin user:password for the readonly role, same host/port/db.
        prefix, rest = self.DATABASE_URL.split("://", 1)
        _, host_and_db = rest.split("@", 1)
        return f"{prefix}://{self.READONLY_DB_USER}:{self.READONLY_DB_PASSWORD}@{host_and_db}"


settings = Settings()
