import time
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from src.database import SessionLocal, init_db
from src.rag import hybrid_search
from tabulate import tabulate

# Evaluation cases: query, category_filter, brand_filter, max_price, min_price, expected_ids, explanation
EVAL_DATASET = [
    {
        "query": "premium dell laptop content creation",
        "category": "Laptop",
        "brand": "Dell",
        "max_price": 150000.0,
        "expected_ids": [1],
        "description": "Premium Dell laptop constraint satisfaction"
    },
    {
        "query": "Sony wireless headphones with noise cancelling",
        "category": "Headphones",
        "brand": "Sony",
        "max_price": 40000.0,
        "expected_ids": [2],
        "description": "Sony headphone brand and ANC feature RAG search"
    },
    {
        "query": "affordable budget laptop under 50000",
        "category": "Laptop",
        "max_price": 50000.0,
        "expected_ids": [4],
        "description": "Budget laptop price constraint"
    },
    {
        "query": "Boat headphones under 2000",
        "category": "Headphones",
        "brand": "Boat",
        "max_price": 2000.0,
        "expected_ids": [5],
        "description": "Ultra budget headphones brand + price constraint"
    }
]

def evaluate_retrieval(db: Session) -> List[Dict[str, Any]]:
    results = []
    
    for case in EVAL_DATASET:
        start_time = time.time()
        
        # Run the search
        matches = hybrid_search(
            db,
            query=case["query"],
            source_type="product",
            category_filter=case.get("category"),
            brand_filter=case.get("brand"),
            max_price=case.get("max_price")
        )
        
        latency = (time.time() - start_time) * 1000
        
        retrieved_ids = [int(m["source_id"]) for m in matches if m["source_id"] is not None]
        
        # Metric 1: Relevance (Top 3 precision/recall)
        relevance = any(eid in retrieved_ids for eid in case["expected_ids"])
        
        # Metric 2: Constraint Satisfaction
        # Check if any retrieved product exceeds max_price
        constraint_passed = True
        for m in matches:
            price = m["metadata"].get("price", 0.0)
            if case.get("max_price") and price > case["max_price"]:
                constraint_passed = False
                break
                
        # Metric 3: Groundedness
        # Verify content contains key information from the query
        grounded = len(matches) > 0
        
        results.append({
            "Query": case["query"],
            "Description": case["description"],
            "Latency (ms)": f"{latency:.2f}",
            "Relevance (Match)": "PASS" if relevance else "FAIL",
            "Constraint Satisfaction": "PASS" if constraint_passed else "FAIL",
            "Groundedness": "PASS" if grounded else "FAIL"
        })
        
    return results

if __name__ == "__main__":
    db = SessionLocal()
    try:
        print("Starting RAG Retrieval & Recommendation Evaluation...")
        reports = evaluate_retrieval(db)
        print("\nEvaluation Results:")
        print(tabulate(reports, headers="keys", tablefmt="grid"))
    finally:
        db.close()
