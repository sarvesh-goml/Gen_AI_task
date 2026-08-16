"""
src/core/memory.py - JARVIS's Memory layer: short-term, long-term, and episodic.

- Short-term: the current conversation, capped so it doesn't grow unbounded (a light
  version of the course's context-window-management lesson).
- Long-term: durable facts, persisted to a JSON file so they survive across sessions -
  backed conceptually by "the vector DB", simplified here to a flat JSON store for the demo.
- Episodic: see src.core.tools.ACTION_LOG for a live example - a structured log of what happened.
"""

import json
import os

from config.settings import BASE_DIR

MEMORY_FILE = os.path.join(BASE_DIR, "long_term_memory.json")
MAX_SHORT_TERM_TURNS = 8  # keep the last N user/assistant turns in context


class ShortTermMemory:
    def __init__(self):
        self.turns = []  # list of {"role": ..., "content": ...}

    def add(self, role, content):
        self.turns.append({"role": role, "content": content})
        if len(self.turns) > MAX_SHORT_TERM_TURNS * 2:
            self.turns = self.turns[-MAX_SHORT_TERM_TURNS * 2:]

    def as_messages(self):
        return list(self.turns)

    def clear(self):
        self.turns = []


def _load_long_term():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_long_term(data):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def remember(key: str, value: str) -> str:
    """Persist a durable fact JARVIS should remember across sessions."""
    data = _load_long_term()
    data[key] = value
    _save_long_term(data)
    return f"Remembered: {key} = {value}"


def recall(key: str) -> str:
    data = _load_long_term()
    return data.get(key, "(nothing on file for that)")


def recall_all() -> dict:
    return _load_long_term()
