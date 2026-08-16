"""
src/core/llm.py - thin wrapper around the Groq API.
"""

import os
from groq import Groq
from config.settings import settings

class GroqUnavailable(RuntimeError):
    pass

def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY") or settings.GROQ_API_KEY
    if not api_key:
        raise GroqUnavailable(
            "GROQ_API_KEY is not set. Please add it to your environment or .env file."
        )
    return Groq(api_key=api_key)

def chat(messages, temperature=0.4, model=None):
    """messages: list of {"role": "system"|"user"|"assistant", "content": str}
    Returns the assistant's reply text."""
    model = model or os.environ.get("GROQ_MODEL") or settings.GROQ_MODEL
    try:
        client = get_groq_client()
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise GroqUnavailable(f"Error communicating with Groq: {e}")

def groq_alive():
    """Best-effort health check used by the Streamlit sidebar."""
    api_key = os.environ.get("GROQ_API_KEY") or settings.GROQ_API_KEY
    if not api_key:
        return False
    try:
        client = Groq(api_key=api_key)
        model = os.environ.get("GROQ_MODEL") or settings.GROQ_MODEL
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            timeout=3,
        )
        return True
    except Exception:
        return False
