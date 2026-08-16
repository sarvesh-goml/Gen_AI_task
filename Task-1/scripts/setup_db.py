"""
scripts/setup_db.py - creates the fleet-ops tables AND the least-privilege read-only
role that src/nl2sql/pipeline.py actually executes generated SQL as (see
src/db/database.py's module docstring for why there are two engines/roles at all).

Idempotent - safe to run against a brand-new database or one that already has the
tables/role from a previous run.

Run from the project root (Postgres must already be up, e.g. `docker compose up -d postgres`):
    python -m scripts.setup_db
"""

import re
import sys

from sqlalchemy import text

from config.settings import settings
from src.db.database import engine
from src.db.models import Base

ROLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def main():
    role = settings.READONLY_DB_USER
    if not ROLE_NAME_RE.match(role):
        sys.exit(f"READONLY_DB_USER '{role}' is not a safe SQL identifier - aborting.")
    password_literal = settings.READONLY_DB_PASSWORD.replace("'", "''")

    Base.metadata.create_all(engine)
    print(f"Tables ensured: {', '.join(t.name for t in Base.metadata.sorted_tables)}")

    with engine.begin() as conn:
        db_name = conn.execute(text("SELECT current_database()")).scalar()

        exists = conn.execute(text("SELECT 1 FROM pg_roles WHERE rolname = :r"), {"r": role}).first()
        if exists:
            conn.execute(text(f"ALTER ROLE {role} WITH LOGIN PASSWORD '{password_literal}'"))
            print(f"Read-only role '{role}' already existed - password refreshed.")
        else:
            conn.execute(text(f"CREATE ROLE {role} WITH LOGIN PASSWORD '{password_literal}'"))
            print(f"Created read-only role: {role}")

        conn.execute(text(f'GRANT CONNECT ON DATABASE "{db_name}" TO {role}'))
        conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {role}"))
        for table in Base.metadata.sorted_tables:
            conn.execute(text(f"GRANT SELECT ON {table.name} TO {role}"))
        print(f"Granted SELECT-only access on {len(Base.metadata.sorted_tables)} fleet tables to '{role}'.")

    print("\nDatabase is up to date.")


if __name__ == "__main__":
    main()
