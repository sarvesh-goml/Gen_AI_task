"""
src/db/database.py - SQLAlchemy engines for the structured fleet-ops PostgreSQL DB.

Two engines, deliberately different privilege levels:
- `engine` - the admin/owner connection. Used ONLY by scripts/setup_db.py (create tables,
  create the readonly role) and scripts/seed_db.py (populate rows). Never used to run
  LLM-generated SQL.
- `readonly_engine` - connects as the least-privilege `jarvis_readonly` role (SELECT-only
  grants on the 4 fleet tables, created by scripts/setup_db.py). This is what
  src/nl2sql/pipeline.py actually executes generated SQL against - the real safety
  boundary, independent of src/nl2sql/guard.py's text-level checks.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.settings import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

readonly_engine = create_engine(settings.readonly_database_url(), pool_pre_ping=True)


def db_reachable(use_readonly=True) -> bool:
    """Best-effort health check used by the API's /health endpoint."""
    target = readonly_engine if use_readonly else engine
    try:
        with target.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        return True
    except Exception:
        return False
