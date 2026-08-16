"""
tests/unit/test_chunking.py - regression tests for src/core/chunking.py. Pure logic, no
Qdrant/Ollama/Postgres needed. Includes a regression test for a real bug caught during
development: the HTML protocol manual's own explanatory comment contained the literal
substrings "<section>" and "</ul>" as prose, which briefly confused the section splitter
into merging the comment into the first protocol's chunk.
"""

from src.core.chunking import INGESTION_PLAN, build_all_chunks


def test_build_all_chunks_covers_every_file_with_no_empty_chunks():
    chunks = build_all_chunks()
    assert len(chunks) > 0
    for text, meta in chunks:
        assert text.strip(), f"empty chunk produced for {meta}"
        assert meta["doc_type"]
        assert meta["source"]
        assert meta["strategy"]


def test_every_ingestion_plan_entry_produces_at_least_one_chunk():
    for chunk_fn, filename, doc_type in INGESTION_PLAN:
        chunks = chunk_fn(filename, doc_type)
        assert len(chunks) > 0, f"{filename} via {chunk_fn.__name__} produced zero chunks"


def test_html_section_chunking_does_not_leak_the_explanatory_comment():
    chunks = [c for c in build_all_chunks() if c[1]["doc_type"] == "protocol_manual"]
    assert len(chunks) == 3
    titles = {meta["title"] for _text, meta in chunks}
    assert titles == {"Protocol: Web-Out", "Protocol: Clean Sweep", "Protocol: Iron Spider Override"}
    for text, _meta in chunks:
        assert "Chunking note" not in text
        assert "<" not in text and ">" not in text


def test_markdown_header_chunking_keeps_heading_with_its_body():
    chunks = [c for c in build_all_chunks() if c[1]["doc_type"] == "mission_debriefs"]
    assert len(chunks) == 4
    for text, meta in chunks:
        assert meta["title"] in text
        assert "Lessons learned" in text


def test_csv_row_chunking_binds_values_to_column_headers():
    chunks = [c for c in build_all_chunks() if c[1]["doc_type"] == "maintenance_log"]
    assert len(chunks) == 5
    first_text, _meta = chunks[0]
    assert "Stark Suit" in first_text
    assert "Happy Hogan" in first_text


def test_json_record_chunking_produces_one_chunk_per_array_element():
    chunks = [c for c in build_all_chunks() if c[1]["doc_type"] == "allies_directory"]
    assert len(chunks) == 4
    names = {meta["name"] for _text, meta in chunks}
    assert "Ned Leeds" in names
