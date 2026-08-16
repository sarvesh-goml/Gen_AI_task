"""
scripts/ingest.py - run this once (or any time you edit the documents/ folder) to build
JARVIS's knowledge base: Ingestion & Chunking -> Embeddings -> Vector DB & Indexing.

Run from the project root:
    python -m scripts.ingest
"""

import os
import sys

# Allow running this file directly (python scripts/ingest.py) as well as via -m.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.chunking import build_all_chunks
from src.core.vector_store import build_collection


def main():
    print("Step 1/2 - Ingestion & Chunking (one strategy per JARVIS knowledge type):")
    chunks = build_all_chunks()
    print(f"\nTotal chunks: {len(chunks)}")

    print("\nStep 2/2 - Embedding + Vector DB indexing (Qdrant, local/embedded, HNSW index)...")
    count = build_collection(chunks)
    print(f"Indexed {count} chunks into Qdrant collection at ./qdrant_data")
    print("\nJARVIS's knowledge base is ready. Run: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
