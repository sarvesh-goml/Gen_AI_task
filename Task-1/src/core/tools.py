"""
src/core/tools.py - Spider-Man AI's agentic Tools layer.
"""

from datetime import datetime

from src.core.vector_store import retrieve
from src.core.rag import format_context
from src.nl2sql.pipeline import answer as nl2sql_answer
import src.core.memory as memory_module

ACTION_LOG = []


def tool_check_suit_status(suit: str = "all") -> str:
    """Read-only tool. Risk: none. Looks up suit diagnostics from the knowledge base."""
    query = f"suit diagnostics status {suit}"
    chunks = retrieve(query, top_k=4, doc_type_filter="suit_diagnostics")
    if not chunks:
        return "No diagnostic records found."
    return "\n".join(c["text"] for c in chunks)


def tool_check_gadget_status(gadget: str = "all") -> str:
    """Read-only tool. Risk: none. Looks up gadget diagnostics from the knowledge base."""
    query = f"gadget diagnostics status {gadget}"
    chunks = retrieve(query, top_k=4, doc_type_filter="suit_diagnostics")
    if not chunks:
        return "No diagnostic records found."
    return "\n".join(c["text"] for c in chunks)


def tool_send_alert(message: str) -> str:
    """Reversible, low-stakes action. Simulated - just logs and returns a confirmation."""
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] ALERT SENT: {message}"
    ACTION_LOG.append(entry)
    print(f"\n  >> {entry}\n")
    return f"Alert sent successfully: '{message}'"


def tool_schedule_reminder(text: str, when: str = "unspecified time") -> str:
    """Reversible, low-stakes action. Simulated - logs and confirms."""
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] REMINDER SET for {when}: {text}"
    ACTION_LOG.append(entry)
    print(f"\n  >> {entry}\n")
    return f"Reminder set for {when}: '{text}'"


def tool_lookup_knowledge_base(query: str) -> str:
    """Read-only tool. Risk: none. Searches the Spider-Man knowledge base."""
    chunks = retrieve(query, top_k=4)
    return format_context(chunks)


def tool_query_spider_man_database(question: str) -> str:
    """Read-only tool. Risk: none. Queries the structured database using NL2SQL."""
    result = nl2sql_answer(question)
    if result.get("error"):
        return f"Could not safely answer from structured records: {result['error']}"
    return f"SQL used: {result['sql']}\n\nAnswer: {result['answer']}"


def tool_view_action_log() -> str:
    """Read-only tool. Shows everything the agent has done this session."""
    if not ACTION_LOG:
        return "No actions taken yet this session."
    return "\n".join(ACTION_LOG)


# Tool registry: name -> (function, description, risk_level, requires_confirmation)
TOOL_REGISTRY = {
    "check_suit_status": {
        "fn": tool_check_suit_status,
        "description": "Look up diagnostic/telemetry data for a suit (e.g. 'Stark Suit', 'Iron Spider', or 'all'). Read-only.",
        "risk": "none",
        "confirm": False,
    },
    "check_gadget_status": {
        "fn": tool_check_gadget_status,
        "description": "Look up diagnostic/telemetry data for gadgets like web shooters or sensors (e.g. 'web shooter', 'cartridge unit', or 'all'). Read-only.",
        "risk": "none",
        "confirm": False,
    },
    "send_alert": {
        "fn": tool_send_alert,
        "description": "Send an alert/notification message. Simulated - prints and logs the alert. Reversible, low-stakes.",
        "risk": "low",
        "confirm": False,
    },
    "schedule_reminder": {
        "fn": tool_schedule_reminder,
        "description": "Schedule a reminder with text and an optional time. Simulated - prints and logs it. Reversible, low-stakes.",
        "risk": "low",
        "confirm": False,
    },
    "lookup_knowledge_base": {
        "fn": tool_lookup_knowledge_base,
        "description": "Search Spider-Man's full knowledge base (humor, moral code, practical support, diagnostics, combat strategy, mission debriefs, allies, protocols) for narrative/procedural information. Read-only.",
        "risk": "none",
        "confirm": False,
    },
    "query_spider_man_database": {
        "fn": tool_query_spider_man_database,
        "description": "Query the structured database (suits, technicians, maintenance history, missions) for precise counts, sums, averages, or joins. Use this instead of lookup_knowledge_base whenever the question needs an exact number or aggregate across records. Read-only.",
        "risk": "none",
        "confirm": False,
    },
    "view_action_log": {
        "fn": tool_view_action_log,
        "description": "View a log of every action taken so far this session. Read-only.",
        "risk": "none",
        "confirm": False,
    },
    "remember_fact": {
        "fn": lambda key, value: memory_module.remember(key, value),
        "description": "Persist a durable fact to long-term memory, e.g. key='allergy', value='shellfish'. Reversible, low-stakes.",
        "risk": "low",
        "confirm": False,
    },
    "recall_fact": {
        "fn": lambda key: memory_module.recall(key),
        "description": "Recall a previously remembered fact by key from long-term memory. Read-only.",
        "risk": "none",
        "confirm": False,
    },
}


def tool_descriptions_block():
    """A formatted block of tool names + descriptions, for the ReAct system prompt."""
    lines = []
    for name, spec in TOOL_REGISTRY.items():
        lines.append(f"- {name}({_signature_hint(name)}): {spec['description']}")
    return "\n".join(lines)


def _signature_hint(name):
    hints = {
        "check_suit_status": "suit='all'",
        "check_gadget_status": "gadget='all'",
        "send_alert": "message",
        "schedule_reminder": "text, when='unspecified time'",
        "lookup_knowledge_base": "query",
        "query_spider_man_database": "question",
        "view_action_log": "",
        "remember_fact": "key, value",
        "recall_fact": "key",
    }
    return hints.get(name, "")


def run_tool(name, **kwargs):
    if name not in TOOL_REGISTRY:
        return f"ERROR: no such tool '{name}'. Available tools: {', '.join(TOOL_REGISTRY.keys())}"
    return TOOL_REGISTRY[name]["fn"](**kwargs)
