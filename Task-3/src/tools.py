import json
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from src.services import ProductService, CartService, WishlistService, OrderService
from src.rag import hybrid_search

# Define the schema of our tools to expose to OpenAI
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search products in the catalog using semantic query and exact filters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Semantic query describing the product."},
                    "category": {"type": "string", "description": "Filter by specific category name."},
                    "brand": {"type": "string", "description": "Filter by brand name."},
                    "min_price": {"type": "number", "description": "Minimum price constraint."},
                    "max_price": {"type": "number", "description": "Maximum price constraint."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "Retrieve full details of a specific product by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer", "description": "The product ID."}
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search store policies, FAQs, buying guides, shipping/returns/warranty info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query about store policies or guides."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_products",
            "description": "Compare features, specifications, prices, and pros/cons of multiple products side-by-side.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of product IDs to compare."
                    }
                },
                "required": ["product_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_reviews",
            "description": "Retrieve reviews and ratings for a product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer", "description": "Product ID."}
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cart",
            "description": "Get current shopping cart content and totals.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": "Add a product to the cart. Requires confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer", "description": "Product ID."},
                    "quantity": {"type": "integer", "description": "Quantity to add.", "default": 1}
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_from_cart",
            "description": "Remove a product from the cart. Requires confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer", "description": "Product ID."}
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_cart_quantity",
            "description": "Change the quantity of a product in the cart. Requires confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer", "description": "Product ID."},
                    "quantity": {"type": "integer", "description": "New quantity."}
                },
                "required": ["product_id", "quantity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_wishlist",
            "description": "Add a product to the wishlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "integer", "description": "Product ID."}
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_coupon",
            "description": "Apply a coupon code (e.g. WELCOME10, SAVE500) to the cart. Requires confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coupon_code": {"type": "string", "description": "The coupon code."}
                },
                "required": ["coupon_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "checkout",
            "description": "Create a pending order and review summary. High-risk, requires explicit confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coupon_code": {"type": "string", "description": "Optional coupon code."}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "Finalize and place the order. High-risk, requires explicit confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer", "description": "Order ID returned by checkout."}
                },
                "required": ["order_id"]
            }
        }
    }
]

# We will implement functions that map exactly to the tools.
# Some tools require user approval. If the tool is called WITHOUT a `confirmed=True` parameter passed from the executor, 
# it returns a request for confirmation.

def execute_tool(
    name: str,
    args: Dict[str, Any],
    db: Session,
    user_id: str,
    confirmed: bool = False
) -> Dict[str, Any]:
    """Routes and executes the specified tool with arguments."""
    
    # Low-risk tools - execute automatically
    if name == "search_products":
        query = args.get("query", "")
        category = args.get("category")
        brand = args.get("brand")
        min_price = args.get("min_price")
        max_price = args.get("max_price")
        
        # Hybrid search combines pgvector and structured filters
        results = hybrid_search(
            db, 
            query=query, 
            source_type="product", 
            category_filter=category, 
            brand_filter=brand, 
            min_price=min_price, 
            max_price=max_price
        )
        return {"status": "success", "results": results}
        
    elif name == "get_product":
        product_id = args.get("product_id")
        product = ProductService.get_product(db, product_id)
        if not product:
            return {"status": "error", "message": f"Product with ID {product_id} not found."}
        return {
            "status": "success",
            "product": {
                "id": product.id,
                "name": product.name,
                "category": product.category,
                "brand": product.brand,
                "price": product.price,
                "rating": product.rating,
                "availability": product.availability,
                "specifications": product.specifications,
                "features": product.features,
                "description": product.description,
                "warranty": product.warranty
            }
        }
        
    elif name == "search_knowledge_base":
        query = args.get("query", "")
        # Search documents in the knowledge base (policies and buying guides)
        results = hybrid_search(db, query=query, limit=3)
        # Filter only non-product chunks or all chunks
        kb_results = [r for r in results if r["source_type"] in ("policy", "buying_guide")]
        return {"status": "success", "results": kb_results if kb_results else results}
        
    elif name == "compare_products":
        product_ids = args.get("product_ids", [])
        comparison = []
        for pid in product_ids:
            p = ProductService.get_product(db, pid)
            if p:
                comparison.append({
                    "id": p.id,
                    "name": p.name,
                    "price": p.price,
                    "brand": p.brand,
                    "rating": p.rating,
                    "specifications": p.specifications,
                    "features": p.features,
                    "availability": p.availability,
                    "warranty": p.warranty
                })
        return {"status": "success", "comparison": comparison}
        
    elif name == "get_product_reviews":
        product_id = args.get("product_id")
        reviews = ProductService.get_product_reviews(db, product_id)
        return {
            "status": "success",
            "reviews": [
                {
                    "rating": r.user_rating,
                    "comment": r.comment,
                    "date": r.review_date.isoformat()
                } for r in reviews
            ]
        }
        
    elif name == "get_cart":
        summary = CartService.get_cart_summary(db, user_id)
        return {"status": "success", "cart": summary}
        
    elif name == "add_to_wishlist":
        product_id = args.get("product_id")
        try:
            item = WishlistService.add_to_wishlist(db, user_id, product_id)
            return {"status": "success", "message": f"Product {product_id} added to wishlist."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # Medium and High Risk tools - check confirmation
    medium_high_risk_actions = {
        "add_to_cart": "medium",
        "remove_from_cart": "medium",
        "update_cart_quantity": "medium",
        "apply_coupon": "medium",
        "checkout": "high",
        "place_order": "high"
    }
    
    if name in medium_high_risk_actions:
        if not confirmed:
            return {
                "status": "requires_confirmation",
                "action_type": name,
                "args": args,
                "message": f"Confirmation required for: {name.replace('_', ' ').title()}."
            }
            
        # If confirmed, proceed
        try:
            if name == "add_to_cart":
                product_id = args.get("product_id")
                qty = args.get("quantity", 1)
                CartService.add_to_cart(db, user_id, product_id, qty)
                return {"status": "success", "message": f"Successfully added product {product_id} to cart."}
                
            elif name == "remove_from_cart":
                product_id = args.get("product_id")
                removed = CartService.remove_from_cart(db, user_id, product_id)
                if removed:
                    return {"status": "success", "message": f"Removed product {product_id} from cart."}
                return {"status": "error", "message": "Product not found in cart."}
                
            elif name == "update_cart_quantity":
                product_id = args.get("product_id")
                qty = args.get("quantity")
                CartService.update_cart_quantity(db, user_id, product_id, qty)
                return {"status": "success", "message": f"Updated product {product_id} quantity to {qty}."}
                
            elif name == "apply_coupon":
                coupon_code = args.get("coupon_code")
                summary = CartService.get_cart_summary(db, user_id, coupon_code)
                if summary["applied_coupon"] == coupon_code.upper().strip():
                    return {"status": "success", "message": f"Coupon {coupon_code} applied successfully.", "cart": summary}
                return {"status": "error", "message": "Invalid or inapplicable coupon."}
                
            elif name == "checkout":
                coupon_code = args.get("coupon_code")
                order = OrderService.checkout(db, user_id, coupon_code)
                return {
                    "status": "success",
                    "message": "Checkout initiated successfully.",
                    "order_id": order.id,
                    "total_price": order.total_price,
                    "status_description": f"Please confirm order ID {order.id} for amount ₹{order.total_price}."
                }
                
            elif name == "place_order":
                order_id = args.get("order_id")
                order = OrderService.place_order(db, user_id, order_id)
                return {
                    "status": "success",
                    "message": f"Order {order.id} placed successfully!",
                    "order_id": order.id
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return {"status": "error", "message": f"Unknown tool: {name}"}
