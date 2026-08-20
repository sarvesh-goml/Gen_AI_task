import time
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from src.config import settings
from src.database import EmbeddingChunk
from fastembed import TextEmbedding

# Initialize local embedding model using fastembed (defaults to BAAI/bge-small-en-v1.5)
embedding_model = TextEmbedding(model_name=settings.EMBEDDING_MODEL_NAME)

def get_embedding(text_to_embed: str) -> List[float]:
    """Generates text embedding locally using fastembed."""
    if not text_to_embed or not text_to_embed.strip():
        return [0.0] * 384
    
    embeddings = list(embedding_model.embed([text_to_embed]))
    emb = embeddings[0]
    if hasattr(emb, "tolist"):
        return emb.tolist()
    return list(emb)

def chunk_markdown_section(content: str, max_chunk_size: int = 1000) -> List[str]:
    """Simple section-based chunking. Splits by markdown headers if possible."""
    lines = content.split("\n")
    chunks = []
    current_chunk = []
    current_size = 0

    for line in lines:
        if line.startswith("#") and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_size = len(line)
        else:
            current_chunk.append(line)
            current_size += len(line) + 1
            if current_size >= max_chunk_size:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_size = 0

    if current_chunk:
        chunks.append("\n".join(current_chunk))
    
    return [c.strip() for c in chunks if c.strip()]

def add_document_to_vector_db(
    db: Session, 
    content: str, 
    source_type: str, 
    source_id: Optional[str] = None, 
    metadata: Optional[Dict[str, Any]] = None
):
    """Chunks a document, generates embeddings, and saves to embedding_chunks table."""
    chunks = chunk_markdown_section(content)
    for chunk in chunks:
        embedding = get_embedding(chunk)
        new_chunk = EmbeddingChunk(
            content=chunk,
            embedding=embedding,
            metadata_info=metadata or {},
            source_type=source_type,
            source_id=source_id
        )
        db.add(new_chunk)
    db.commit()

def hybrid_search(
    db: Session,
    query: str,
    limit: int = 5,
    source_type: Optional[str] = None,
    category_filter: Optional[str] = None,
    brand_filter: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Performs hybrid search:
    1. Generates query embedding.
    2. Performs pgvector cosine distance search.
    3. Merges and filters based on exact metadata.
    """
    query_emb = get_embedding(query)
    
    conditions = []
    params = {"query_emb": query_emb, "limit": limit}
    
    if source_type:
        conditions.append("source_type = :source_type")
        params["source_type"] = source_type
        
    if category_filter:
        conditions.append("CAST(metadata_info->>'category' AS VARCHAR) ILIKE :category")
        params["category"] = f"%{category_filter}%"
        
    if brand_filter:
        conditions.append("CAST(metadata_info->>'brand' AS VARCHAR) ILIKE :brand")
        params["brand"] = f"%{brand_filter}%"
        
    if min_price is not None:
        conditions.append("CAST(metadata_info->>'price' AS DOUBLE PRECISION) >= :min_price")
        params["min_price"] = min_price
        
    if max_price is not None:
        conditions.append("CAST(metadata_info->>'price' AS DOUBLE PRECISION) <= :max_price")
        params["max_price"] = max_price

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)
        
    sql = text(f"""
        SELECT id, content, source_type, source_id, metadata_info, 1 - (embedding <=> :query_emb) AS similarity
        FROM embedding_chunks
        {where_clause}
        ORDER BY embedding <=> :query_emb
        LIMIT :limit
    """)
    
    results = db.execute(sql, params).fetchall()
    
    output = []
    for r in results:
        output.append({
            "id": r.id,
            "content": r.content,
            "source_type": r.source_type,
            "source_id": r.source_id,
            "metadata": r.metadata_info,
            "similarity": float(r.similarity) if r.similarity is not None else 0.0
        })
    return output
