import os
import json
import datetime
from sqlalchemy.orm import Session
from src.database import Product, ProductReview, EmbeddingChunk, init_db
from src.rag import get_embedding

# Let's read product and policy data from files in the knowledge directory if they exist.
# Otherwise, use fallback inline data.

PRODUCTS_FALLBACK = [
    {
        "id": 1,
        "name": "Dell XPS 15",
        "category": "Laptop",
        "brand": "Dell",
        "price": 120000.0,
        "rating": 4.5,
        "availability": True,
        "specifications": {
            "processor": "Intel Core Ultra 7",
            "ram": "16 GB",
            "storage": "1 TB SSD",
            "display": "15.6-inch OLED",
            "gpu": "NVIDIA RTX 4050"
        },
        "features": ["Content creation ready", "Lightweight design", "Excellent battery life"],
        "description": "Premium laptop designed for professional and creative workloads. High performance meets elegance.",
        "warranty": "1 Year Premium Support"
    },
    {
        "id": 2,
        "name": "Sony WH-1000XM5",
        "category": "Headphones",
        "brand": "Sony",
        "price": 29999.0,
        "rating": 4.8,
        "availability": True,
        "specifications": {
            "type": "Over-ear",
            "connectivity": "Wireless Bluetooth 5.2",
            "battery_life": "30 hours",
            "noise_cancelling": "Industry-leading ANC"
        },
        "features": ["Active Noise Cancellation", "Speak-to-chat", "Multipoint connection"],
        "description": "Industry leading noise-canceling headphones with premium sound and comfort. Perfect for travel.",
        "warranty": "1 Year Brand Warranty"
    },
    {
        "id": 3,
        "name": "Samsung Galaxy S24 Ultra",
        "category": "Smartphone",
        "brand": "Samsung",
        "price": 129999.0,
        "rating": 4.9,
        "availability": True,
        "specifications": {
            "processor": "Snapdragon 8 Gen 3",
            "ram": "12 GB",
            "storage": "256 GB",
            "display": "6.8-inch Dynamic AMOLED",
            "camera": "200 MP Quad Camera"
        },
        "features": ["Built-in S Pen", "Galaxy AI tools", "Titanium frame"],
        "description": "The ultimate Android experience with cutting-edge camera zoom, advanced performance, and integrated AI capabilities.",
        "warranty": "1 Year Manufacturer Warranty"
    },
    {
        "id": 4,
        "name": "Lenovo IdeaPad Slim 3",
        "category": "Laptop",
        "brand": "Lenovo",
        "price": 38000.0,
        "rating": 4.0,
        "availability": True,
        "specifications": {
            "processor": "Intel Core i3 12th Gen",
            "ram": "8 GB",
            "storage": "512 GB SSD",
            "display": "15.6-inch FHD"
        },
        "features": ["Affordable productivity", "Privacy shutter camera", "Dolby Audio Speakers"],
        "description": "Reliable budget laptop for school, online learning, and basic productivity tasks.",
        "warranty": "1 Year Onsite Warranty"
    },
    {
        "id": 5,
        "name": "Boat Rockerz 450",
        "category": "Headphones",
        "brand": "Boat",
        "price": 1499.0,
        "rating": 3.9,
        "availability": True,
        "specifications": {
            "type": "On-ear",
            "connectivity": "Bluetooth 5.0",
            "battery_life": "15 hours"
        },
        "features": ["Extra bass", "Foldable design", "Voice assistant integration"],
        "description": "Very popular affordable wireless headphones with punchy bass and long battery play time.",
        "warranty": "1 Year Brand Warranty"
    }
]

REVIEWS = [
    {"product_id": 1, "user_rating": 5.0, "comment": "Amazing display and very fast performance. Best laptop I have owned!"},
    {"product_id": 1, "user_rating": 4.0, "comment": "Excellent quality but gets a bit warm under high workloads."},
    {"product_id": 2, "user_rating": 5.0, "comment": "Active noise cancellation is magical. Highly recommended for travelers."},
    {"product_id": 3, "user_rating": 5.0, "comment": "Camera quality is incredible, especially the zoom. AI features are fun to use."},
    {"product_id": 4, "user_rating": 4.0, "comment": "Great value for money. Perfect for everyday usage like browsing and documents."}
]

KNOWLEDGE_DOCS_FALLBACK = [
    {
        "content": """# Store Shipping Policy
We offer free shipping on all orders over ₹2000. For orders under ₹2000, a flat shipping fee of ₹100 is charged.
Delivery Times:
- Metro Cities: 2 to 4 business days.
- Other Cities: 5 to 7 business days.
Same-day delivery is available in select cities (Mumbai, Bangalore, Delhi) for orders placed before 12 PM with a surcharge of ₹150.""",
        "source_type": "policy",
        "metadata": {"title": "Shipping Policy", "category": "policies"}
    },
    {
        "content": """# Return and Refund Policy
We want you to love your purchases. If you are not satisfied, you can return or exchange any item within 15 days of delivery.
Exceptions:
- Electronics and gadgets must be returned in their original, unopened packaging with all seals intact.
- Items marked as 'Final Sale' cannot be returned or refunded.
Refund process takes 5 to 7 business days after the product arrives and passes inspections at our warehouse.""",
        "source_type": "policy",
        "metadata": {"title": "Return & Refund Policy", "category": "policies"}
    },
    {
        "content": """# Warranty Claims FAQ
All products purchased from our store come with standard brand warranty.
How to claim:
1. Locate your store invoice in your profile or email.
2. Visit the nearest authorized service center for the respective brand (Dell, Sony, Samsung, Lenovo, Boat).
3. Present the invoice to initiate warranty claims.
If you face issues, reach out to support@shopassistant.com and our agent will assist in forwarding the claim.""",
        "source_type": "policy",
        "metadata": {"title": "Warranty FAQ", "category": "policies"}
    },
    {
        "content": """# Laptop Buying Guide 2026
Choosing a laptop depends heavily on your usage:
1. Productivity/Office: Focus on battery life, comfortable keyboard, and portability. Look for Intel Core i5/Ultra 5 or AMD Ryzen 5 with 16 GB RAM. (e.g. Dell XPS, Lenovo Thinkpad).
2. Creative/Gaming: Dedicated GPU is a must (e.g. NVIDIA RTX 40-series). 16GB or 32GB RAM, OLED or high refresh rate screen.
3. Budget/Students: Focus on reliability. Intel Core i3 or Ryzen 3 with 8GB RAM is sufficient for standard tasks.""",
        "source_type": "buying_guide",
        "metadata": {"title": "Laptop Buying Guide", "category": "guides"}
    }
]

def load_products_from_knowledge() -> list:
    knowledge_dir = "/Users/sivakarthick/shop/knowledge"
    if not os.path.exists(knowledge_dir):
        return PRODUCTS_FALLBACK

    products = []
    # Try to scan files in folders
    pid = 1
    for root, dirs, files in os.walk(knowledge_dir):
        for file in files:
            if file.endswith(".md") and "policy" not in file and "faq" not in file and "guide" not in file:
                path = os.path.join(root, file)
                try:
                    with open(path, "r") as f:
                        lines = f.read().split("\n")
                    # Simple parse
                    pdata = {"id": pid, "specifications": {}, "features": []}
                    desc_mode = False
                    spec_mode = False
                    feat_mode = False
                    description_lines = []
                    
                    for line in lines:
                        if not line.strip():
                            continue
                        if line.startswith("Product:"):
                            pdata["name"] = line.replace("Product:", "").strip()
                        elif line.startswith("Category:"):
                            pdata["category"] = line.replace("Category:", "").strip()
                        elif line.startswith("Brand:"):
                            pdata["brand"] = line.replace("Brand:", "").strip()
                        elif line.startswith("Price:"):
                            pdata["price"] = float(line.replace("Price:", "").strip())
                        elif line.startswith("Rating:"):
                            pdata["rating"] = float(line.replace("Rating:", "").strip())
                        elif line.startswith("Availability:"):
                            pdata["availability"] = line.replace("Availability:", "").strip().lower() == "true"
                        elif line.startswith("Warranty:"):
                            pdata["warranty"] = line.replace("Warranty:", "").strip()
                        elif line.startswith("Description:"):
                            desc_mode = True
                            spec_mode = False
                            feat_mode = False
                        elif line.startswith("Specifications:"):
                            desc_mode = False
                            spec_mode = True
                            feat_mode = False
                        elif line.startswith("Features:"):
                            desc_mode = False
                            spec_mode = False
                            feat_mode = True
                        elif desc_mode:
                            description_lines.append(line.strip())
                        elif spec_mode and line.startswith("-"):
                            parts = line[1:].split(":")
                            if len(parts) >= 2:
                                pdata["specifications"][parts[0].strip().lower()] = parts[1].strip()
                        elif feat_mode and line.startswith("-"):
                            pdata["features"].append(line[1:].strip())
                            
                    pdata["description"] = " ".join(description_lines)
                    products.append(pdata)
                    pid += 1
                except Exception as e:
                    print(f"Error parsing {file}: {e}")
                    
    return products if products else PRODUCTS_FALLBACK

def load_policies_from_knowledge() -> list:
    knowledge_dir = "/Users/sivakarthick/shop/knowledge/policies"
    if not os.path.exists(knowledge_dir):
        return KNOWLEDGE_DOCS_FALLBACK

    docs = []
    for file in os.listdir(knowledge_dir):
        if file.endswith(".md"):
            path = os.path.join(knowledge_dir, file)
            try:
                with open(path, "r") as f:
                    content = f.read()
                docs.append({
                    "content": content,
                    "source_type": "policy",
                    "metadata": {"title": file.replace(".md", "").replace("_", " ").title(), "category": "policies"}
                })
            except Exception as e:
                print(f"Error reading {file}: {e}")
    return docs if docs else KNOWLEDGE_DOCS_FALLBACK

def seed_database_and_rag(db: Session):
    # Clear existing data safely
    db.query(ProductReview).delete()
    db.query(Product).delete()
    db.query(EmbeddingChunk).delete()
    db.commit()

    products = load_products_from_knowledge()
    policies = load_policies_from_knowledge()

    print(f"Seeding {len(products)} products...")
    for p in products:
        prod = Product(
            id=p["id"],
            name=p["name"],
            category=p["category"],
            brand=p["brand"],
            price=p["price"],
            rating=p["rating"],
            availability=p["availability"],
            specifications=p["specifications"],
            features=p["features"],
            description=p["description"],
            warranty=p.get("warranty", "1 Year Brand Warranty")
        )
        db.add(prod)
        
        # Create product description chunk for RAG
        content = f"""Product: {p['name']}
Category: {p['category']}
Brand: {p['brand']}
Price: ₹{p['price']}
Rating: {p['rating']}/5
Description: {p['description']}
Specifications:
{chr(10).join([f"- {k.replace('_', ' ').title()}: {v}" for k, v in p['specifications'].items()])}
Features:
{chr(10).join([f"- {f}" for f in p['features']])}
Warranty: {p.get('warranty', '')}
"""
        emb = get_embedding(content)
        chunk = EmbeddingChunk(
            content=content,
            embedding=emb,
            metadata_info={"category": p["category"], "brand": p["brand"], "price": p["price"], "rating": p["rating"]},
            source_type="product",
            source_id=str(p["id"])
        )
        db.add(chunk)
        
    db.commit()

    print("Seeding reviews...")
    for r in REVIEWS:
        review = ProductReview(
            product_id=r["product_id"],
            user_rating=r["user_rating"],
            comment=r["comment"],
            review_date=datetime.datetime.utcnow()
        )
        db.add(review)
    db.commit()

    print(f"Seeding {len(policies)} policies and guides...")
    for d in policies:
        emb = get_embedding(d["content"])
        chunk = EmbeddingChunk(
            content=d["content"],
            embedding=emb,
            metadata_info=d["metadata"],
            source_type=d["source_type"]
        )
        db.add(chunk)
    db.commit()
    print("Database seeding completed.")

if __name__ == "__main__":
    from src.database import SessionLocal
    init_db()
    db = SessionLocal()
    try:
        seed_database_and_rag(db)
    finally:
        db.close()
