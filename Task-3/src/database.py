from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Text, JSON, Numeric
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import text
import datetime
from pgvector.sqlalchemy import Vector
from src.config import settings

Base = declarative_base()

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False, index=True)
    brand = Column(String, nullable=False, index=True)
    price = Column(Float, nullable=False)
    rating = Column(Float, default=0.0)
    availability = Column(Boolean, default=True)
    specifications = Column(JSON, default=dict)
    features = Column(JSON, default=list)  # list of strings
    description = Column(Text, nullable=True)
    warranty = Column(String, nullable=True)

    reviews = relationship("ProductReview", back_populates="product", cascade="all, delete-orphan")

class ProductReview(Base):
    __tablename__ = "product_reviews"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    user_rating = Column(Float, nullable=False)
    comment = Column(Text, nullable=True)
    review_date = Column(DateTime, default=datetime.datetime.utcnow)

    product = relationship("Product", back_populates="reviews")

class EmbeddingChunk(Base):
    __tablename__ = "embedding_chunks"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(384))  # fastembed BAAI/bge-small-en-v1.5 uses 384 dimensions
    metadata_info = Column(JSON, default=dict)  # named metadata_info to avoid sqlalchemy keyword conflicts
    source_type = Column(String, nullable=False)  # 'product', 'policy', 'buying_guide'
    source_id = Column(String, nullable=True)     # Reference to product ID or other document reference

class CartItem(Base):
    __tablename__ = "cart_items"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, default=1)

    product = relationship("Product")

class WishlistItem(Base):
    __tablename__ = "wishlist_items"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)

    product = relationship("Product")

class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    status = Column(String, default="pending_confirmation")  # pending_confirmation, placed, cancelled
    total_price = Column(Float, nullable=False)
    coupon_applied = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")

class OrderItem(Base):
    __tablename__ = "order_items"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)

    order = relationship("Order", back_populates="items")
    product = relationship("Product")

class InteractionLog(Base):
    __tablename__ = "interaction_logs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    latency_ms = Column(Float, nullable=False)
    tokens_used = Column(Integer, default=0)
    cost = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Engine and Session Setup
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    # Make sure pgvector extension exists
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
