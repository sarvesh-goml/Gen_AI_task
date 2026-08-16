"""
src/core/chunking.py - different chunking strategies for different JARVIS knowledge types.

This directly demonstrates the course's Ingestion & Chunking lesson along two axes at
once: content STRUCTURE differs (a joke vs. a procedure needs a different split), and
file FORMAT differs (plain text, Markdown, JSON, CSV, and HTML each need their own
parser before you can even think about chunking). Each function below returns a list of
(text, metadata) chunks.
"""

import csv
import json
import os
import re

from config.settings import settings

DOCUMENTS_DIR = settings.DOCUMENTS_DIR


def _read(filename):
    path = os.path.join(DOCUMENTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _strip_comments(text):
    """Remove the '# ...' explanatory comment lines at the top of each doc file."""
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))


def chunk_by_sentence(filename, doc_type):
    """One-liner / sentence-level chunking - for humor & personality lines.
    Each quip must stand alone; mixing several into one chunk would dilute retrieval."""
    text = _strip_comments(_read(filename))
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return [(line, {"doc_type": doc_type, "source": filename, "strategy": "sentence"}) for line in lines]


def chunk_by_paragraph(filename, doc_type):
    """Paragraph-level semantic chunking - for moral code & practical support.
    Each rule/routine needs its full reasoning kept intact in one chunk."""
    text = _strip_comments(_read(filename))
    paragraphs = [p.strip().replace("\n", " ") for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [(p, {"doc_type": doc_type, "source": filename, "strategy": "paragraph"}) for p in paragraphs]


def chunk_by_record(filename, doc_type):
    """Structured, per-record chunking - for suit diagnostics/telemetry.
    Never separate a value from its unit, never merge two records into one chunk."""
    text = _strip_comments(_read(filename))
    records = [l.strip() for l in text.splitlines() if l.strip()]
    return [(r, {"doc_type": doc_type, "source": filename, "strategy": "record"}) for r in records]


def chunk_by_procedure(filename, doc_type):
    """Recursive / structure-aware chunking - for combat strategy.
    Each numbered procedure (with all its steps) stays together as ONE chunk, so a
    generic fixed-size splitter can never cut a step in half or drop a threshold value."""
    text = _strip_comments(_read(filename))
    blocks = re.split(r"\n\s*\n(?=Procedure:)", text.strip())
    chunks = []
    for block in blocks:
        block = block.strip()
        if block:
            title_match = re.match(r"Procedure:\s*(.+)", block)
            title = title_match.group(1).strip() if title_match else "Untitled procedure"
            chunks.append((block.replace("\n", " "), {"doc_type": doc_type, "source": filename, "strategy": "procedure", "title": title}))
    return chunks


def chunk_by_markdown_header(filename, doc_type):
    """Markdown header-aware chunking - for mission debriefs.
    Splits on '## ' section boundaries so a mission's outcome and lessons-learned never
    get separated from its heading, no matter how many bullets sit underneath it."""
    text = _read(filename)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)  # strip markdown-style comments
    sections = re.split(r"\n(?=## )", text.strip())
    chunks = []
    for section in sections:
        section = section.strip()
        if not section or section.startswith("# "):
            continue  # the lone H1 title (no leading "## ") isn't a mission of its own
        title_match = re.match(r"##\s*(.+)", section)
        title = title_match.group(1).strip() if title_match else "Untitled mission"
        flat = re.sub(r"\s+", " ", section).strip()
        chunks.append((flat, {"doc_type": doc_type, "source": filename, "strategy": "markdown_header", "title": title}))
    return chunks


def chunk_by_json_record(filename, doc_type):
    """Structured JSON chunking - for the allies directory.
    Parses with json.load (never regex/text-split JSON) so one array element always
    becomes exactly one chunk, with every field represented and none silently dropped."""
    records = json.loads(_read(filename))
    chunks = []
    for record in records:
        name = record.get("name", "Unknown")
        fields = "; ".join(f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in record.items() if k != "name")
        flat = f"{name} - {fields}"
        chunks.append((flat, {"doc_type": doc_type, "source": filename, "strategy": "json_record", "name": name}))
    return chunks


def chunk_by_csv_row(filename, doc_type):
    """Row-level CSV chunking - for the maintenance log.
    Parses with csv.DictReader (never naive line-splitting) so each value stays bound to
    its column header even if a field itself happens to contain a comma."""
    path = os.path.join(DOCUMENTS_DIR, filename)
    chunks = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            flat = (
                f"On {row['Date']}, {row['Suit']}'s {row['Component']} had an issue: "
                f"{row['Issue']}. Resolution: {row['Resolution']}. Technician: {row['Technician']}."
            )
            chunks.append((flat, {"doc_type": doc_type, "source": filename, "strategy": "csv_row", "date": row["Date"]}))
    return chunks


def chunk_by_html_section(filename, doc_type):
    """HTML tag-aware chunking - for the protocol manual.
    Splits on <section> boundaries and strips markup, so retrieval returns clean prose
    (never raw tags) and a section is never cut off mid-<ul>."""
    text = _read(filename)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)  # strip HTML comments first
    sections = re.findall(r"<section>(.*?)</section>", text, flags=re.DOTALL)
    chunks = []
    for section in sections:
        title_match = re.search(r"<h2>(.*?)</h2>", section, flags=re.DOTALL)
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else "Untitled protocol"
        clean = re.sub(r"<[^>]+>", " ", section)
        clean = re.sub(r"\s+", " ", clean).strip()
        chunks.append((clean, {"doc_type": doc_type, "source": filename, "strategy": "html_section", "title": title}))
    return chunks


# The actual ingestion plan: which file uses which strategy, matching the course's cheat sheet.
INGESTION_PLAN = [
    (chunk_by_sentence, "humor_style.txt", "humor"),
    (chunk_by_paragraph, "moral_code.txt", "moral_code"),
    (chunk_by_paragraph, "practical_support.txt", "practical_support"),
    (chunk_by_record, "suit_diagnostics.txt", "suit_diagnostics"),
    (chunk_by_procedure, "combat_strategy.txt", "combat_strategy"),
    (chunk_by_markdown_header, "mission_debriefs.md", "mission_debriefs"),
    (chunk_by_json_record, "allies_directory.json", "allies_directory"),
    (chunk_by_csv_row, "maintenance_log.csv", "maintenance_log"),
    (chunk_by_html_section, "protocol_manual.html", "protocol_manual"),
]


def build_all_chunks():
    """Runs the full ingestion plan and returns every (text, metadata) chunk across all doc types."""
    all_chunks = []
    for chunk_fn, filename, doc_type in INGESTION_PLAN:
        chunks = chunk_fn(filename, doc_type)
        all_chunks.extend(chunks)
        print(f"  {filename:<22} -> {chunk_fn.__name__:<20} -> {len(chunks)} chunks")
    return all_chunks
