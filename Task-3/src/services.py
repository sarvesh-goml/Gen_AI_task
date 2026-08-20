from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from src.database import Product, ProductReview, CartItem, WishlistItem, Order, OrderItem

class ProductService:
    @staticmethod
    def search_products(
        db: Session,
        query: Optional[str] = None,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        available_only: bool = True
    ) -> List[Product]:
        """Structured product query matching exact filters."""
        q = db.query(Product)
        
        if category:
            q = q.filter(Product.category.ilike(category))
        if brand:
            q = q.filter(Product.brand.ilike(brand))
        if min_price is not None:
            q = q.filter(Product.price >= min_price)
        if max_price is not None:
            q = q.filter(Product.price <= max_price)
        if available_only:
            q = q.filter(Product.availability == True)
            
        if query:
            # Simple keyword search fallback/addition
            search_pattern = f"%{query}%"
            q = q.filter(
                or_(
                    Product.name.ilike(search_pattern),
                    Product.description.ilike(search_pattern)
                )
            )
            
        return q.all()

    @staticmethod
    def get_product(db: Session, product_id: int) -> Optional[Product]:
        return db.query(Product).filter(Product.id == product_id).first()

    @staticmethod
    def get_product_reviews(db: Session, product_id: int) -> List[ProductReview]:
        return db.query(ProductReview).filter(ProductReview.product_id == product_id).all()


class CartService:
    @staticmethod
    def get_cart(db: Session, user_id: str) -> List[CartItem]:
        return db.query(CartItem).filter(CartItem.user_id == user_id).all()

    @staticmethod
    def add_to_cart(db: Session, user_id: str, product_id: int, quantity: int = 1) -> CartItem:
        # Check if product is available
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product or not product.availability:
            raise ValueError("Product is not available or does not exist.")
            
        item = db.query(CartItem).filter(
            and_(CartItem.user_id == user_id, CartItem.product_id == product_id)
        ).first()
        
        if item:
            item.quantity += quantity
        else:
            item = CartItem(user_id=user_id, product_id=product_id, quantity=quantity)
            db.add(item)
            
        db.commit()
        db.refresh(item)
        return item

    @staticmethod
    def remove_from_cart(db: Session, user_id: str, product_id: int) -> bool:
        item = db.query(CartItem).filter(
            and_(CartItem.user_id == user_id, CartItem.product_id == product_id)
        ).first()
        if item:
            db.delete(item)
            db.commit()
            return True
        return False

    @staticmethod
    def update_cart_quantity(db: Session, user_id: str, product_id: int, quantity: int) -> Optional[CartItem]:
        if quantity <= 0:
            CartService.remove_from_cart(db, user_id, product_id)
            return None
            
        item = db.query(CartItem).filter(
            and_(CartItem.user_id == user_id, CartItem.product_id == product_id)
        ).first()
        if item:
            item.quantity = quantity
            db.commit()
            db.refresh(item)
            return item
        return None

    @staticmethod
    def get_cart_summary(db: Session, user_id: str, coupon_code: Optional[str] = None) -> Dict[str, Any]:
        items = CartService.get_cart(db, user_id)
        subtotal = sum(item.product.price * item.quantity for item in items)
        
        discount = 0.0
        applied_coupon = None
        if coupon_code:
            code = coupon_code.upper().strip()
            if code == "WELCOME10":
                discount = subtotal * 0.10
                applied_coupon = "WELCOME10"
            elif code == "SAVE500" and subtotal >= 2000:
                discount = 500.0
                applied_coupon = "SAVE500"
                
        total = max(0.0, subtotal - discount)
        return {
            "items": [
                {
                    "product_id": item.product_id,
                    "name": item.product.name,
                    "price": item.product.price,
                    "quantity": item.quantity,
                    "total": item.product.price * item.quantity
                } for item in items
            ],
            "subtotal": subtotal,
            "discount": discount,
            "applied_coupon": applied_coupon,
            "total": total
        }


class WishlistService:
    @staticmethod
    def get_wishlist(db: Session, user_id: str) -> List[WishlistItem]:
        return db.query(WishlistItem).filter(WishlistItem.user_id == user_id).all()

    @staticmethod
    def add_to_wishlist(db: Session, user_id: str, product_id: int) -> WishlistItem:
        # Check if already in wishlist
        item = db.query(WishlistItem).filter(
            and_(WishlistItem.user_id == user_id, WishlistItem.product_id == product_id)
        ).first()
        if not item:
            item = WishlistItem(user_id=user_id, product_id=product_id)
            db.add(item)
            db.commit()
            db.refresh(item)
        return item

    @staticmethod
    def remove_from_wishlist(db: Session, user_id: str, product_id: int) -> bool:
        item = db.query(WishlistItem).filter(
            and_(WishlistItem.user_id == user_id, WishlistItem.product_id == product_id)
        ).first()
        if item:
            db.delete(item)
            db.commit()
            return True
        return False


class OrderService:
    @staticmethod
    def checkout(db: Session, user_id: str, coupon_code: Optional[str] = None) -> Order:
        """Create a pending order from current cart items, but do not place/finalize it yet."""
        cart_summary = CartService.get_cart_summary(db, user_id, coupon_code)
        if not cart_summary["items"]:
            raise ValueError("Cart is empty.")
            
        # Create a pending order
        order = Order(
            user_id=user_id,
            status="pending_confirmation",
            total_price=cart_summary["total"],
            coupon_applied=cart_summary["applied_coupon"]
        )
        db.add(order)
        db.flush()  # gets order.id
        
        # Move items from cart to order items
        cart_items = CartService.get_cart(db, user_id)
        for item in cart_items:
            order_item = OrderItem(
                order_id=order.id,
                product_id=item.product_id,
                quantity=item.quantity,
                price=item.product.price
            )
            db.add(order_item)
            
        db.commit()
        db.refresh(order)
        return order

    @staticmethod
    def place_order(db: Session, user_id: str, order_id: int) -> Order:
        """Confirm and place a pending order, clearing the cart."""
        order = db.query(Order).filter(
            and_(Order.id == order_id, Order.user_id == user_id)
        ).first()
        
        if not order:
            raise ValueError("Order not found.")
        if order.status != "pending_confirmation":
            raise ValueError(f"Order cannot be placed because it is in '{order.status}' state.")
            
        order.status = "placed"
        
        # Clear Cart
        cart_items = CartService.get_cart(db, user_id)
        for item in cart_items:
            db.delete(item)
            
        db.commit()
        db.refresh(order)
        return order

    @staticmethod
    def cancel_order(db: Session, user_id: str, order_id: int) -> Order:
        order = db.query(Order).filter(
            and_(Order.id == order_id, Order.user_id == user_id)
        ).first()
        if not order:
            raise ValueError("Order not found.")
            
        order.status = "cancelled"
        db.commit()
        db.refresh(order)
        return order
