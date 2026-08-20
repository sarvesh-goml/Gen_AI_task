import pytest
from unittest.mock import MagicMock, patch
from src.rag import chunk_markdown_section, get_embedding, hybrid_search

def test_chunk_markdown_section():
    content = """# Header 1
This is paragraph 1.
# Header 2
This is paragraph 2.
- Point 1
- Point 2"""
    chunks = chunk_markdown_section(content, max_chunk_size=100)
    assert len(chunks) == 2
    assert chunks[0].startswith("# Header 1")
    assert chunks[1].startswith("# Header 2")

@patch("src.rag.embedding_model")
def test_get_embedding(mock_embedding_model):
    # Mock fastembed response
    mock_embedding_model.embed.return_value = [[0.1] * 384]

    emb = get_embedding("Hello test")
    assert len(emb) == 384
    assert emb[0] == 0.1
    mock_embedding_model.embed.assert_called_once_with(["Hello test"])
