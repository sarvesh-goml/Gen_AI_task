"""
src/api/session_store.py - in-memory, per-(session_id, mode) short-term conversation
memory for the API. Deliberately a plain process-local dict, not Redis/a DB - this is a
single-instance training demo, not a service that needs to survive a restart or scale
past one worker (see src/api/main.py's single-worker enforcement for why that matters).
"""

import threading

from src.core.memory import ShortTermMemory

_lock = threading.Lock()
_sessions: dict[str, ShortTermMemory] = {}


def get_session(key: str) -> ShortTermMemory:
    with _lock:
        if key not in _sessions:
            _sessions[key] = ShortTermMemory()
        return _sessions[key]


def clear_session(key: str) -> None:
    with _lock:
        _sessions.pop(key, None)
