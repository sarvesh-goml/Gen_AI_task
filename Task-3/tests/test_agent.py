import pytest
from unittest.mock import MagicMock
from src.tools import execute_tool
from src.agent import estimate_cost

def test_estimate_cost():
    cost = estimate_cost("llama-3.3-70b-versatile", 1000, 500)
    # 1000 * 0.59/1M + 500 * 0.79/1M = 0.00059 + 0.000395 = 0.000985
    assert abs(cost - 0.000985) < 1e-9

def test_execute_tool_low_risk():
    db = MagicMock()
    # Mocking low-risk product lookup
    db.query.return_value.filter.return_value.first.return_value = None
    
    res = execute_tool("get_product", {"product_id": 999}, db, "user123")
    assert res["status"] == "error"  # returns product not found

def test_execute_tool_medium_high_risk_without_confirmation():
    db = MagicMock()
    
    # checkout is high risk
    res = execute_tool("checkout", {}, db, "user123", confirmed=False)
    assert res["status"] == "requires_confirmation"
    assert res["action_type"] == "checkout"

    # add_to_cart is medium risk
    res = execute_tool("add_to_cart", {"product_id": 1, "quantity": 1}, db, "user123", confirmed=False)
    assert res["status"] == "requires_confirmation"
    assert res["action_type"] == "add_to_cart"
