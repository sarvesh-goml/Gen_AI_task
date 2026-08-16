"""
src/core/agent.py - Spider-Man AI's Agentic AI layer: Brain (LLM) + Tools + Memory + Planning &
Reasoning, wired together using the ReAct architecture (Think -> Act -> Observe).
"""

import json
import re

from config.settings import settings
from src.core.llm import chat
from src.core.tools import TOOL_REGISTRY, tool_descriptions_block, run_tool
from src.core.memory import ShortTermMemory

SPIDEY_SYSTEM_PROMPT = settings.SPIDEY_SYSTEM_PROMPT

MAX_ITERATIONS = 5

REACT_INSTRUCTIONS = f"""
You are Spider-Man AI operating in AGENTIC mode - you can call tools to take real action,
not just reply. Use the ReAct pattern: reason about what to do, optionally call ONE tool,
observe the result, and repeat until you can give a Final Answer.

Available tools:
{tool_descriptions_block()}

STRICT OUTPUT FORMAT - follow this exactly, one block per turn:
Thought: <your reasoning about what to do next>
Action: <a tool name from the list above, OR "none" if you're ready to answer>
Action Input: <a JSON object of arguments for the tool, e.g. {{"message": "..."}}, or {{}} if Action is "none">

When you have enough information and/or have taken the necessary action(s), respond with:
Thought: <final reasoning>
Action: none
Action Input: {{}}
Final Answer: <your in-character Spider-Man AI reply to Peter, confirming what was done>

Rules:
- Only call ONE tool per turn.
- Never skip the Thought line.
- Action Input must be valid JSON (use {{}} for tools with no arguments).
"""

BLOCK_RE = re.compile(
    r"Thought:\s*(?P<thought>.*?)\s*"
    r"Action:\s*(?P<action>.*?)\s*"
    r"Action Input:\s*(?P<action_input>\{.*?\}|\{\})"
    r"(?:.*?Final Answer:\s*(?P<final>.*))?",
    re.DOTALL,
)


def _parse_step(text):
    match = BLOCK_RE.search(text.strip())
    if not match:
        return {"thought": "(format not followed)", "action": "none", "action_input": {}, "final": text.strip()}
    action = match.group("action").strip()
    try:
        action_input = json.loads(match.group("action_input").strip())
    except json.JSONDecodeError:
        action_input = {}
    return {
        "thought": match.group("thought").strip(),
        "action": action,
        "action_input": action_input,
        "final": (match.group("final") or "").strip() or None,
    }


def run_agent(query, short_term: ShortTermMemory, verbose=True):
    """Runs the ReAct loop for one user query.

    Returns (final_answer, trace) where trace is a list of step dicts:
      {"thought": str, "action": str, "action_input": dict, "observation": str|None}
    """
    messages = [{"role": "system", "content": SPIDEY_SYSTEM_PROMPT + "\n" + REACT_INSTRUCTIONS}]
    messages.extend(short_term.as_messages())
    messages.append({"role": "user", "content": query})

    trace = []

    for i in range(MAX_ITERATIONS):
        raw = chat(messages, temperature=0.2)
        step = _parse_step(raw)

        if verbose:
            print(f"\n  [Planning step {i + 1}] Thought: {step['thought']}")

        if step["final"]:
            trace.append({"thought": step["thought"], "action": "none", "action_input": {}, "observation": None, "final": True})
            return step["final"], trace

        action = step["action"]
        entry = {"thought": step["thought"], "action": action, "action_input": step["action_input"], "observation": None, "final": False}

        if action and action.lower() != "none" and action in TOOL_REGISTRY:
            if verbose:
                print(f"  [Planning step {i + 1}] Action: {action}({step['action_input']})")
            try:
                observation = run_tool(action, **step["action_input"])
            except Exception as e:
                observation = f"ERROR running tool '{action}': {e}"
            if verbose:
                print(f"  [Planning step {i + 1}] Observation: {observation}")
            entry["observation"] = observation
            trace.append(entry)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Observation: {observation}\n\nContinue with the next Thought/Action, or give your Final Answer."})
        else:
            trace.append(entry)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Please provide your Final Answer now."})

    return "Peter, I've hit my planning limit for this request - let's try rephrasing it or breaking it down.", trace
