from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from src.database import get_db, init_db
from src.schemas import ChatRequest, ChatResponse, ConfirmRequest
from src.agent import run_agent_loop, confirm_pending_action
from src.services import CartService
from scripts.seed import seed_database_and_rag
import uuid

app = FastAPI(
    title="AI Shopping Assistant API",
    description="Backend API for the RAG-powered single-agent AI Shopping Assistant.",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    init_db()

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "AI Shopping Assistant"}

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        # Run agent loop
        result = run_agent_loop(
            db=db,
            message=request.message,
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        
        # Get cart summary to return to the user if any
        cart_summary = CartService.get_cart_summary(db, request.user_id)
        
        suggested_actions = []
        if result.get("status") == "requires_confirmation":
            suggested_actions = ["Approve action", "Deny action"]
        else:
            suggested_actions = ["Search headphones", "Compare premium laptops", "View my cart"]
            
        return ChatResponse(
            response=result["response"],
            status=result["status"],
            cart_summary=cart_summary,
            suggested_actions=suggested_actions,
            request_id=result["request_id"],
            pending_action=result.get("pending_action")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/confirm")
def confirm_endpoint(request: ConfirmRequest, db: Session = Depends(get_db)):
    approve = request.confirm_action.lower() == "approve"
    try:
        res = confirm_pending_action(
            db=db,
            action_id=request.pending_action_id,
            approve=approve,
            user_id=request.user_id
        )
        if res.get("status") == "error":
            raise HTTPException(status_code=400, detail=res.get("message"))
            
        cart_summary = CartService.get_cart_summary(db, request.user_id)
        return {
            "status": "success",
            "message": "Action processed.",
            "execution_result": res.get("result") or res.get("message"),
            "cart_summary": cart_summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/seed")
def seed_endpoint(db: Session = Depends(get_db)):
    try:
        seed_database_and_rag(db)
        return {"status": "success", "message": "Database and RAG indexing seeded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
