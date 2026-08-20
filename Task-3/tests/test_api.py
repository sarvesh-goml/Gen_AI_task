import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "AI Shopping Assistant"}

@patch("src.main.run_agent_loop")
@patch("src.main.CartService")
def test_chat_endpoint(mock_cart_service, mock_agent_loop):
    # Setup mocks
    mock_agent_loop.return_value = {
        "response": "Here is Sony WH-1000XM5.",
        "status": "success",
        "request_id": "req-123",
        "conversation_id": "conv-123"
    }
    mock_cart_service.get_cart_summary.return_value = {"items": [], "total": 0.0}
    
    response = client.post(
        "/api/chat",
        json={"message": "I want Sony headphones", "user_id": "user1"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Here is Sony WH-1000XM5."
    assert data["status"] == "success"
    assert data["request_id"] == "req-123"
