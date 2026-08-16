"""
src/ui/cli.py - command-line entry point for the Spider-Man AI demo.
"""

import argparse
import sys
import uuid

import requests

from config.settings import settings

API_BASE_URL = settings.API_BASE_URL

BANNER = r"""
  ___  ___  _ ___  ___  ___  __  __  _   _    _   _ 
 / __|| _ \| |   \| __|| _ \|  \/  |/_\ | |  | | | |
 \__ \|  _/| | |) | _||   /| |\/| / _ \| |__| |_| |
 |___/|_|  |_|___/|___||_|_\|_|  |_/_/ \_\____\___/ 
"""


def _post_chat(endpoint, session_id, query):
    resp = requests.post(
        f"{API_BASE_URL}/api/v1/chat/{endpoint}",
        json={"session_id": session_id, "query": query},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def run_rag_mode():
    print(BANNER)
    print("Spider-Man AI (RAG mode) online. I can answer from my knowledge base - I will not take actions.")
    print("Type 'exit' to quit.\n")
    session_id = str(uuid.uuid4())

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            print("Spider-Man AI: Heading back to Queens! Catch you later, Peter.")
            break
        if not query:
            continue
        try:
            data = _post_chat("rag", session_id, query)
        except requests.exceptions.RequestException as e:
            print(f"\n[SETUP ISSUE] Could not reach the API at {API_BASE_URL}: {e}\n")
            continue

        print(f"\nSpider-Man AI: {data['reply']}\n")
        if data.get("citations"):
            print("  (retrieved from: " + ", ".join(data["citations"]) + ")\n")


def run_agent_mode():
    print(BANNER)
    print("Spider-Man AI (AGENTIC mode) online. I can look things up (including the structured "
          "database) AND take action now.")
    print("Type 'exit' to quit.\n")
    session_id = str(uuid.uuid4())

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            print("Spider-Man AI: Heading back to Queens! Catch you later, Peter.")
            break
        if not query:
            continue
        try:
            data = _post_chat("agent", session_id, query)
        except requests.exceptions.RequestException as e:
            print(f"\n[SETUP ISSUE] Could not reach the API at {API_BASE_URL}: {e}\n")
            continue

        print(f"\nSpider-Man AI: {data['reply']}\n")


def main():
    parser = argparse.ArgumentParser(description="Spider-Man AI demo chatbot (RAG or Agentic mode)")
    parser.add_argument("--mode", choices=["rag", "agent"], default="rag", help="Which capability to demo")
    args = parser.parse_args()

    if args.mode == "rag":
        run_rag_mode()
    else:
        run_agent_mode()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSpider-Man AI: Powering down.")
        sys.exit(0)
