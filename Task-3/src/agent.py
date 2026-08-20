import json
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from src.config import settings
from src.database import InteractionLog
from src.tools import TOOL_SCHEMAS, execute_tool
from groq import Groq

# Initialize native Groq Client
groq_client = Groq(api_key=settings.GROQ_API_KEY if settings.GROQ_API_KEY else "mock-key-for-import")

# In-memory session store for simplicity (simulates short-term memory)
# Format: {session_id: [messages]}
MEMORY_STORE: Dict[str, List[Dict[str, Any]]] = {}

# In-memory pending actions store for confirmation flows
# Format: {pending_action_id: {"user_id": user_id, "tool_name": tool_name, "args": args}}
PENDING_ACTIONS: Dict[str, Dict[str, Any]] = {}

SYSTEM_PROMPT = """You are a professional, helpful AI Shopping Assistant.
Your goal is to assist users in discovering products, comparing them, managing their cart, and checking out.

Guidelines:
1. Always use RAG tools (`search_products`, `search_knowledge_base`) to answer queries about products, specifications, stock, shipping, returns, and policies.
2. Explain recommendations clearly: focus on price, ratings, features, advantages, and limitations based on the fetched data. Do not make up product attributes.
3. Be helpful and natural. Maintain session context (e.g. remember brand preferences, budget limits mentioned earlier).
4. If a tool requires confirmation (like checkout or placing an order), explain that you need user confirmation and prompt them to confirm it.
"""

def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimates cost based on token counts (using Llama 3 on Groq rates as fallback)."""
    # standard llama-3.3-70b pricing: ~$0.59 / 1M input tokens, ~$0.79 / 1M output tokens
    input_rate = 0.59 / 1_000_000
    output_rate = 0.79 / 1_000_000
    return (prompt_tokens * input_rate) + (completion_tokens * output_rate)

def run_agent_loop(
    db: Session,
    message: str,
    user_id: str,
    conversation_id: Optional[str] = None
) -> Dict[str, Any]:
    """Runs the agent tool-calling loop, returning a final response or a confirmation request."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        
    if conversation_id not in MEMORY_STORE:
        MEMORY_STORE[conversation_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        
    # Append the user's message to context
    MEMORY_STORE[conversation_id].append({"role": "user", "content": message})
    
    messages = list(MEMORY_STORE[conversation_id])
    
    max_steps = 5
    step = 0
    pending_action = None
    response_status = "success"
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    while step < max_steps:
        step += 1
        try:
            # We use standard tool calling with Groq
            response = groq_client.chat.completions.create(
                model=settings.GROQ_MODEL_NAME,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto"
            )
        except Exception as e:
            # Safe fallback response
            err_msg = f"Sorry, I encountered an error communicating with the model: {str(e)}"
            return {
                "response": err_msg,
                "status": "error",
                "request_id": request_id,
                "conversation_id": conversation_id
            }
            
        choice = response.choices[0]
        message_obj = choice.message
        
        # Track usage
        if response.usage:
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens
            
        # Append assistant's response object
        # Groq choice.message has a custom structure, convert it to dict if necessary,
        # but to keep it simple we can append the dict version or standard object.
        # We append a dict representation to avoid serialization issues
        msg_dict = {"role": "assistant", "content": message_obj.content}
        if message_obj.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in message_obj.tool_calls
            ]
        messages.append(msg_dict)
        
        if not message_obj.tool_calls:
            # No tools to call, we have our final text answer
            MEMORY_STORE[conversation_id] = messages
            break
            
        # Execute tool calls
        for tool_call in message_obj.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            # Execute tool
            result = execute_tool(tool_name, tool_args, db, user_id, confirmed=False)
            
            if result.get("status") == "requires_confirmation":
                # Intercept execution, generate a pending action ID, and stop the loop
                action_id = str(uuid.uuid4())
                PENDING_ACTIONS[action_id] = {
                    "user_id": user_id,
                    "tool_name": tool_name,
                    "args": tool_args,
                    "conversation_id": conversation_id
                }
                
                pending_action = {
                    "pending_action_id": action_id,
                    "action_type": tool_name,
                    "args": tool_args,
                    "message": result.get("message")
                }
                response_status = "requires_confirmation"
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps({"status": "requires_confirmation", "pending_action_id": action_id})
                })
                break
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(result)
                })
                
        if pending_action:
            MEMORY_STORE[conversation_id] = messages
            break

    final_text = messages[-1].get("content") or messages[-2].get("content") or "How can I assist you further?"
    if not isinstance(final_text, str):
        final_text = getattr(final_text, "content", None) or "I have processed your request."
        
    latency = (time.time() - start_time) * 1000
    cost = estimate_cost(settings.GROQ_MODEL_NAME, total_prompt_tokens, total_completion_tokens)
    
    # Log the interaction
    log = InteractionLog(
        request_id=request_id,
        prompt=message,
        response=final_text,
        latency_ms=latency,
        tokens_used=total_prompt_tokens + total_completion_tokens,
        cost=cost
    )
    db.add(log)
    db.commit()
    
    return {
        "response": final_text,
        "status": response_status,
        "request_id": request_id,
        "conversation_id": conversation_id,
        "pending_action": pending_action
    }

def confirm_pending_action(
    db: Session,
    action_id: str,
    approve: bool,
    user_id: str
) -> Dict[str, Any]:
    """Executes a previously paused tool call once user approves it, or cancels it."""
    action = PENDING_ACTIONS.get(action_id)
    if not action:
        return {"status": "error", "message": "Pending action not found or expired."}
        
    if action["user_id"] != user_id:
        return {"status": "error", "message": "Unauthorized for this action."}
        
    if not approve:
        del PENDING_ACTIONS[action_id]
        return {"status": "cancelled", "message": f"Action {action['tool_name']} has been cancelled."}
        
    tool_name = action["tool_name"]
    tool_args = action["args"]
    conversation_id = action["conversation_id"]
    
    result = execute_tool(tool_name, tool_args, db, user_id, confirmed=True)
    
    del PENDING_ACTIONS[action_id]
    
    if conversation_id in MEMORY_STORE:
        MEMORY_STORE[conversation_id].append({
            "role": "system",
            "content": f"User approved and successfully executed tool '{tool_name}' with args {json.dumps(tool_args)}. Result: {json.dumps(result)}"
        })
        
    return {
        "status": "success",
        "result": result
    }
