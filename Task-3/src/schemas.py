from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    message: str
    user_id: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    status: str  # "success", "requires_confirmation", "error"
    cart_summary: Optional[Dict[str, Any]] = None
    suggested_actions: Optional[List[str]] = None
    request_id: str
    pending_action: Optional[Dict[str, Any]] = None

class ConfirmRequest(BaseModel):
    user_id: str
    action_type: str  # "checkout", "place_order", "apply_coupon", "add_to_cart", "remove_from_cart", "update_cart_quantity"
    confirm_action: str  # "approve" or "deny"
    pending_action_id: str
