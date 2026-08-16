"""
src/nl2sql/schema_introspection.py - builds the schema text handed to the LLM from
SQLAlchemy's own metadata (src/db/models.py), not a hand-maintained string. This means
the prompt schema can never silently drift out of sync with the real database - if a
column is renamed in models.py, the next call picks it up automatically.
"""

from src.db.models import Base


def get_schema_description() -> str:
    lines = ["You have access to a PostgreSQL database with the following tables:", ""]
    for table in Base.metadata.sorted_tables:
        columns = []
        for col in table.columns:
            piece = f"{col.name} {col.type}"
            if col.foreign_keys:
                target = next(iter(col.foreign_keys)).target_fullname
                piece += f" (references {target})"
            columns.append(piece)
        lines.append(f"Table: {table.name}")
        for c in columns:
            lines.append(f"  - {c}")
        lines.append("")
    return "\n".join(lines)
