"""
Act 3 — LLMOps: run the bakery.

A thin FastAPI wrapper around the fine-tuned (or distilled) model, following
Slide 27's advice: keep the API layer thin (routing + validation only) and
push real logic into a separate module. Configuration comes from environment
variables, never hardcoded, so the same image can point at different model
paths in different environments.

Run locally:
    uvicorn app:app --reload --port 8000

Endpoints:
    GET  /health              -- liveness check
    POST /generate             -- {"prompt": "..."} -> {"response": "..."}
"""

import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration (environment variables, not hardcoded -- Slide 27's warning)
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "../fine_tuning/outputs/lora-adapter")
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are Atlas, the friendly virtual travel agent for Wanderlust Travels. "
    "Answer questions warmly, concisely, and accurately about trip planning, "
    "destinations, bookings, travel requirements, and itineraries. "
    "Keep replies to 2-4 sentences.",
)
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "1024"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "150"))
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "true").lower() == "true"

app = FastAPI(title="Wanderlust Travels — Travel Assistant API", version="1.0.0")

_model = None
_tokenizer = None


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = None


class GenerateResponse(BaseModel):
    response: str
    latency_seconds: float
    model_path: str


def _load_model():
    """Lazy-load the model on first request so `uvicorn app:app` starts fast
    and /health works even before the (slow) model load completes."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(model)
    _model, _tokenizer = model, tokenizer
    return _model, _tokenizer


@app.get("/health")
def health():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="`prompt` must not be empty")

    try:
        model, tokenizer = _load_model()
    except Exception as exc:  # pragma: no cover - surfaced to the caller
        raise HTTPException(status_code=503, detail=f"Model failed to load: {exc}") from exc

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": req.prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    start = time.time()
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=req.max_new_tokens or MAX_NEW_TOKENS,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    latency = time.time() - start

    generated = outputs[0][inputs.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return GenerateResponse(response=text, latency_seconds=round(latency, 3), model_path=MODEL_PATH)
