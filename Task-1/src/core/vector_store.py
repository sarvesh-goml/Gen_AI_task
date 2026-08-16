"""
src/core/vector_store.py - the Vector DB & Indexing layer.

Uses Qdrant in embedded/local mode (QdrantClient(path=...)) so there is NO server or
Docker requirement - the whole index lives in a folder on disk. Indexing algorithm is
Qdrant's default HNSW, which is exactly what the course recommends for latency-sensitive
retrieval (combat-speed lookups).
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from config.settings import settings

QDRANT_PATH = settings.QDRANT_PATH
COLLECTION_NAME = settings.COLLECTION_NAME
EMBEDDING_MODEL = settings.EMBEDDING_MODEL
TOP_K = settings.TOP_K

_embedder = None
_client = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def get_client():
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH)
    return _client


def build_collection(chunks):
    """chunks: list of (text, metadata) tuples. Embeds and upserts them all into Qdrant."""
    embedder = get_embedder()
    client = get_client()

    vector_size = embedder.get_sentence_embedding_dimension()
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
    )

    texts = [c[0] for c in chunks]
    metadatas = [c[1] for c in chunks]
    vectors = embedder.encode(texts, show_progress_bar=False).tolist()

    points = [
        qmodels.PointStruct(id=i, vector=vectors[i], payload={"text": texts[i], **metadatas[i]})
        for i in range(len(texts))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def collection_ready():
    """True if the knowledge base has already been built (used by the Streamlit sidebar)."""
    try:
        client = get_client()
        return client.collection_exists(COLLECTION_NAME)
    except Exception:
        return False


def retrieve(query, top_k=TOP_K, doc_type_filter=None):
    """Retrieval & Context Injection layer: embeds the query, searches Qdrant, returns
    the top_k chunks (with metadata) most relevant to the query. Optionally filter by
    doc_type (e.g. only search 'combat_strategy') for hybrid-style precision lookups."""
    embedder = get_embedder()
    client = get_client()
    query_vector = embedder.encode(query).tolist()

    query_filter = None
    if doc_type_filter:
        query_filter = qmodels.Filter(
            must=[qmodels.FieldCondition(key="doc_type", match=qmodels.MatchValue(value=doc_type_filter))]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
    ).points

    return [
        {"text": r.payload["text"], "doc_type": r.payload["doc_type"], "score": r.score}
        for r in results
    ]
