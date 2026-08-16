# Spider-Man AI Demo — RAG, Structured-Data & Agentic AI

> A production-shaped reference chatbot combining vector retrieval over multi-format
> documents, a natural-language-to-SQL pipeline over a real relational database, and an
> agentic ReAct loop that chooses between them — built for the RAG & Agentic AI training session,
> using Spider-Man AI as the running example.

Everything runs with **Groq API** for the LLM, `sentence-transformers`
for embeddings, Qdrant (embedded) for the vector DB, and PostgreSQL (via Docker) for the
structured DB.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
   - 2.1 [Vector knowledge base & multi-format chunking](#21-vector-knowledge-base--multi-format-chunking)
   - 2.2 [Structured database & the NL2SQL pipeline](#22-structured-database--the-nl2sql-pipeline)
   - 2.3 [Agentic ReAct loop & the tool registry](#23-agentic-react-loop--the-tool-registry)
   - 2.4 [Memory](#24-memory)
   - 2.5 [The FastAPI service boundary](#25-the-fastapi-service-boundary)
3. [Project Structure](#3-project-structure)
4. [Prerequisites](#4-prerequisites)
5. [Complete Local Setup](#5-complete-local-setup)
6. [Sample Queries](#6-sample-queries)
7. [API Reference](#7-api-reference)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Project Overview

This chatbot answers questions about Spider-Man's world — Peter Parker's suits, allies,
protocols, and missions — by combining **two independent retrieval subsystems** plus an
LLM, orchestrated either directly (RAG mode) or through an agent that picks tools for
itself (Agentic mode):

1. **Vector knowledge base** — 9 documents across 5 file formats (`.txt`, `.md`, `.json`,
   `.csv`, `.html`), each chunked with the strategy appropriate to its structure, embedded
   with `sentence-transformers`, and indexed in Qdrant (embedded/local).
   Good for narrative, procedural, and personality knowledge.
2. **Structured database** — a real PostgreSQL database (`suits`,
   `technicians`, `maintenance_events`, `missions`) queried through a **natural-language-
   to-SQL pipeline**: the LLM writes a query, a safety guard and a least-privilege DB role
   both gate it, and a second LLM call turns the result rows into a spoken answer. Good
   for anything that needs an exact count, sum, average, or join.
3. **Groq** running a fast Llama 3 model, synthesizing retrieved
   context (from either or both subsystems) into a Spider-Man AI response.

**Two modes, switchable from the sidebar:**
- **RAG mode** — Spider-Man AI answers questions grounded in the vector knowledge base only. It
  only replies, never acts.
- **Agentic mode** — Spider-Man AI gets a `TOOL_REGISTRY` (vector search, SQL database
  queries, alerts, reminders, memory) and a ReAct planning loop. The UI shows its full
  Thought → Action → Observation trace, including any SQL it generated, for every reply.

---

## 2. Architecture

```
                    ┌────────────────────┐        ┌────────────────────┐
                    │  Streamlit UI       │        │  CLI                │
                    │  src/ui/streamlit_  │        │  src/ui/cli.py      │
                    │  app.py             │        │                     │
                    └──────────┬──────────┘        └──────────┬──────────┘
                               │  HTTP (both are thin clients - no logic of their own)
                               ▼
                    ┌──────────────────────────────────────────────────┐
                    │            FastAPI service (Port 8000)            │
                    │                 src/api/main.py                   │
                    │  POST /api/v1/chat/rag     -> src.core.rag         │
                    │  POST /api/v1/chat/agent   -> src.core.agent (ReAct)│
                    │  POST /api/v1/sql/query    -> src.nl2sql.pipeline   │
                    │  GET  /health                                      │
                    └───────┬───────────────────┬────────────────────────┘
                            │                    │
              ┌─────────────┘                    └───────────────┐
              ▼                                                   ▼
   ┌─────────────────────┐                          ┌───────────────────────────┐
   │ Vector KB (Qdrant,   │                          │ Structured DB (PostgreSQL) │
   │ embedded, local)     │                          │ suits / technicians /      │
   │ src/core/vector_store│                          │ maintenance_events /       │
   │ src/core/chunking    │                          │ missions                   │
   └──────────┬───────────┘                          └─────────────┬──────────────┘
              │                                                    │
              │              ┌──────────────────────┐              │
              └─────────────►│   Ollama (local LLM)  │◄─────────────┘
                              │   src/core/llm.py     │
                              └──────────────────────┘
```

### 2.1 Vector knowledge base & multi-format chunking

`data/documents/` holds 9 files across 5 formats, each demonstrating a *different*
chunking need — the course's ingestion cheat sheet, made runnable in
`src/core/chunking.py`'s `INGESTION_PLAN`:

| Knowledge type | File | Format | Strategy | Why |
|---|---|---|---|---|
| Humor & personality | `humor_style.txt` | text | Sentence | Each quip must stand alone |
| Moral code | `moral_code.txt` | text | Paragraph | Reasoning must stay intact |
| Practical support | `practical_support.txt` | text | Paragraph | Each routine is self-contained |
| Suit diagnostics | `suit_diagnostics.txt` | text | Structured/per-record | Never separate a value from its unit |
| Combat strategy | `combat_strategy.txt` | text | Recursive/structure-aware | Preserves procedure step order |
| Mission debriefs | `mission_debriefs.md` | Markdown | Header-aware | Splits on `## ` so a mission's lessons-learned never separate from its heading |
| Allies directory | `allies_directory.json` | JSON | Per-record | Parsed with `json.load`, never regex — one array element = one chunk |
| Maintenance log | `maintenance_log.csv` | CSV | Per-row | Parsed with `csv.DictReader` so values stay bound to column headers |
| Protocol manual | `protocol_manual.html` | HTML | Per-section | Splits on `<section>` tags, strips markup |

Two axes are deliberately in play: content **structure** differs (a joke vs. a
procedure), and file **format** differs (Markdown/JSON/CSV/HTML each need their own
parser before chunking can even start — text-splitting raw JSON or HTML is the classic
mistake this catches). `scripts/ingest.py` runs the whole plan and embeds the result.

### 2.2 Structured DB & the NL2SQL pipeline

The vector KB is great at narrative/procedural knowledge and bad at precise aggregates —
"how many times has the Stark Suit needed left-web-shooter repairs" needs an exact COUNT,
not an LLM eyeballing a handful of retrieved chunks. So the operational records
live in a real relational schema instead (`src/db/models.py`):

```
suits ──┬──< maintenance_events >──┬── technicians
        │
        └──< missions
```

| Table | Columns | Rows seeded |
|---|---|---|
| `suits` | mark_name, status, power_core_pct, last_diagnostic_date | 6 |
| `technicians` | name, specialty, years_experience | 6 |
| `maintenance_events` | suit, technician, date, component, issue, resolution, hours, cost | 30 |
| `missions` | suit, date, location, threat_level, duration_min, outcome | 20 |

Seed rows (`src/db/seed_data.py`) are hand-authored, not random, so every benchmark
question in `sample_queries.md` has one verifiable correct answer — and a few facts (the
Stark Suit's left web shooter being flagged 3 times) are kept consistent with
`suit_diagnostics.txt`.

**The NL2SQL pipeline** (`src/nl2sql/`), given a question:

```
Question
    │
    ▼
schema_introspection.get_schema_description()   <- reads live SQLAlchemy metadata,
    │                                               so the prompt schema can never drift
    ▼
generator.generate_sql()   <- LLM writes ONE SELECT statement
    │
    ▼
guard.validate_select_only()   <- rejects non-SELECT / multiple statements / banned
    │                              keywords (DEFENSE IN DEPTH, not the real boundary)
    ▼
readonly_engine.execute()   <- runs as the `jarvis_readonly` Postgres role, which has
    │                          SELECT-only grants on the 4 tables (THE REAL
    │                          BOUNDARY - created by scripts/setup_db.py). Even SQL the
    │                          guard missed cannot write or drop anything, because the
    │                          role it executes as has no privilege to.
    ▼
pipeline._synthesize_answer()   <- second LLM call turns the result rows into a
                                    Spider-Man AI answer, citing "structured records"
```

One retry is allowed: if the guard rejects the first attempt, its rejection reason is fed
back to the LLM for a corrected try before giving up gracefully.

### 2.3 Agentic ReAct loop & the tool registry

`src/core/agent.py` runs a hand-rolled, text-parsed ReAct loop (Thought → Action →
Observation) against `src/core/tools.py`'s `TOOL_REGISTRY`:

| Tool | Backing subsystem | Read-only? |
|---|---|---|
| `lookup_knowledge_base` | Vector KB (§2.1) | Yes |
| `query_spider_man_database` | Structured DB via NL2SQL (§2.2) | Yes |
| `check_suit_status` | Vector KB, filtered to `suit_diagnostics` | Yes |
| `check_gadget_status` | Vector KB, filtered to `suit_diagnostics` | Yes |
| `view_action_log` | Episodic memory (§2.4) | Yes |
| `recall_fact` | Long-term memory (§2.4) | Yes |
| `send_alert`, `schedule_reminder` | Simulated actions | No (low-stakes, reversible) |
| `remember_fact` | Long-term memory (§2.4) | No (low-stakes, reversible) |

The lookup tools are the contrast: `lookup_knowledge_base`
for narrative/procedural questions, `query_spider_man_database` for anything needing an exact
number or a join — each tool's description is written precisely enough for the agent to
pick correctly, and a query that needs both (e.g. "how many web shooter repairs has the
Stark Suit had, and does the combat doc say anything about swinging with a degraded shooter?")
will chain them in one turn. See `sample_queries.md` for a query crafted exactly for this.

### 2.4 Memory

- **Short-term**: the current conversation, capped at the last 8 turns
  (`src/core/memory.ShortTermMemory`), held server-side per `(session_id, mode)` in
  `src/api/session_store.py`.
- **Long-term**: durable facts (`remember_fact`/`recall_fact`), persisted to
  `long_term_memory.json` at the project root, surviving restarts.
- **Episodic**: `src/core/tools.ACTION_LOG` — every simulated action taken this run,
  readable via `view_action_log`.

### 2.5 The FastAPI service boundary

Streamlit and the CLI are **thin HTTP clients** of one FastAPI process — neither has any
RAG/Agentic/NL2SQL logic of its own (`src/api/main.py`). This is a real production
constraint worth knowing: the session store, the action log, and the Qdrant/embedder
singletons are all in-process Python state, not backed by a shared store. `main.py`'s
`_enforce_single_worker()` refuses to start if `--workers N>1` or a
`WEB_CONCURRENCY`/`WORKERS` env var above 1 is set — a misconfigured multi-worker deploy
is caught at startup, not as a confusing intermittent bug later.

---

## 3. Project Structure

```
jarvis_demo/
├── config/
│   └── settings.py              # pydantic-settings - every setting, env-driven
├── src/
│   ├── core/                    # RAG + Agentic logic, UI-agnostic
│   │   ├── chunking.py          # Ingestion & Chunking - 9 strategies across 5 formats
│   │   ├── vector_store.py      # Qdrant embedded client + embeddings + retrieval
│   │   ├── llm.py               # thin wrapper around local Ollama
│   │   ├── rag.py               # Retrieval & Context Injection + Generation
│   │   ├── memory.py            # short-term (ShortTermMemory) + long-term (JSON file)
│   │   ├── tools.py             # TOOL_REGISTRY - one function per agent capability
│   │   └── agent.py             # ReAct loop (Thought/Action/Observation)
│   ├── db/
│   │   ├── database.py          # admin engine + readonly engine (SQLAlchemy)
│   │   ├── models.py            # Suit, Technician, MaintenanceEvent, Mission
│   │   └── seed_data.py         # hand-authored, deterministic seed rows
│   ├── nl2sql/
│   │   ├── schema_introspection.py  # builds prompt schema from live DB metadata
│   │   ├── generator.py         # LLM call: question + schema -> candidate SQL
│   │   ├── guard.py             # SELECT-only / single-statement / banned-keyword checks
│   │   └── pipeline.py          # generate -> guard -> execute -> synthesize answer
│   ├── api/
│   │   ├── main.py              # FastAPI app, startup checks, single-worker enforcement
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── session_store.py     # in-memory per-(session_id, mode) conversation memory
│   │   └── routes/
│   │       ├── health.py        # GET /health
│   │       ├── chat.py          # POST /api/v1/chat/{rag,agent}
│   │       └── sql.py           # POST /api/v1/sql/query
│   └── ui/
│       ├── streamlit_app.py     # chat UI - HTTP client of src/api only
│       └── cli.py               # terminal alternative - same HTTP client pattern
├── scripts/
│   ├── ingest.py                 # build the vector KB
│   ├── setup_db.py               # create fleet tables + the jarvis_readonly role
│   └── seed_db.py                # populate the 4 fleet tables from seed_data.py
├── data/
│   └── documents/                 # the 9 knowledge-base files (§2.1)
├── tests/unit/
│   ├── test_chunking.py          # chunking regression tests (incl. the HTML-comment bug)
│   └── test_nl2sql_guard.py      # SQL-guard accept/reject cases
├── docker-compose.yml             # PostgreSQL only (Qdrant stays embedded/local)
├── requirements.txt
├── .env.example
├── Makefile
├── pytest.ini
├── README.md
└── sample_queries.md              # ready-made prompts covering every concept
```

---

## 4. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | 3.13/3.14 supported |
| [Ollama](https://ollama.com) | Local LLM runtime - no API key |
| Docker Desktop | For the PostgreSQL fleet-ops DB only (Qdrant needs no Docker) |

---

## 5. Complete Local Setup

### Step 1 — Install Ollama and pull a model
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b        # or: ollama pull qwen2.5:3b (lighter machine)
```
If you pull a different model, set `OLLAMA_MODEL` in `.env` (copy from `.env.example`) to
match. Quick check Ollama is alive: `ollama list`.

### Step 2 — Python environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3 — Configure environment
```bash
cp .env.example .env
```
Every value has a working local default — you generally only need to touch
`OLLAMA_MODEL` if you pulled a different one.

### Step 4 — Start PostgreSQL and build the fleet-ops DB
```bash
docker compose up -d postgres
python -m scripts.setup_db     # creates the 4 tables + the read-only jarvis_readonly role
python -m scripts.seed_db      # populates them from src/db/seed_data.py
```
Both scripts are idempotent — safe to re-run any time.

### Step 5 — Build the vector knowledge base
```bash
python -m scripts.ingest
```
Chunks all 9 files in `data/documents/` with the format/structure-appropriate strategy
(§2.1), embeds them, and indexes into a local Qdrant collection at `./qdrant_data/`
(created automatically — no server or Docker needed for this part).

### Step 6 — Run the API
```bash
uvicorn src.api.main:app --reload --port 8000
```
Verify: [http://localhost:8000/health](http://localhost:8000/health) should show
`knowledge_base`, `ollama`, and `postgres` all `true`. Interactive docs at
[http://localhost:8000/docs](http://localhost:8000/docs).

### Step 7 — Run the UI
New terminal:
```bash
source .venv/bin/activate
streamlit run src/ui/streamlit_app.py
```
Use the sidebar to switch between **RAG** and **Agentic** mode, and check the
system-status panel (knowledge base / Ollama / fleet DB / API) if something isn't
responding.

A lightweight terminal alternative is also available:
```bash
python -m src.ui.cli --mode rag      # or --mode agent
```

Or use `make setup` to run steps 2–5 in one shot, then `make run-api` / `make run-ui`.

---

## 6. Sample Queries

See `sample_queries.md` for the full list, including:
- Vector-KB queries proving each chunking strategy (one per file/format in §2.1)
- Fleet-DB queries with known, hand-checkable answers (§2.2's seed data)
- An agentic query crafted to require **both** tools in one turn

---

## 7. API Reference

Full interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Knowledge-base / Ollama / Postgres reachability |
| POST | `/api/v1/chat/rag` | `{session_id, query}` -> grounded reply + citations, RAG mode |
| POST | `/api/v1/chat/agent` | `{session_id, query}` -> reply + full ReAct trace, Agentic mode |
| DELETE | `/api/v1/chat/session/{session_id}?mode=rag\|agent` | Clear that session's short-term memory |
| POST | `/api/v1/sql/query` | `{question}` -> NL2SQL pipeline directly, bypassing the agent |

```bash
curl http://localhost:8000/health

curl -s -X POST http://localhost:8000/api/v1/sql/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many times has the Mark 42 needed left boot thruster repairs?"}' \
  | python3 -m json.tool

curl -s -X POST http://localhost:8000/api/v1/chat/agent \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo-1", "query": "How many maintenance events has the Mark 45 had, and what did they cost in total?"}' \
  | python3 -m json.tool
```

---

## 8. Troubleshooting

- **Sidebar shows "API not reachable"** — start the API first: `uvicorn src.api.main:app --reload --port 8000`.
- **Sidebar shows "Ollama not reachable"** — Ollama isn't running. Try `ollama list`; if
  that fails, reinstall/start Ollama.
- **Sidebar shows "Knowledge base not built yet"** — run `python -m scripts.ingest`.
- **Sidebar shows "Fleet DB not reachable"** — run `docker compose up -d postgres`, then
  `python -m scripts.setup_db` and `python -m scripts.seed_db`.
- **`role "jarvis" does not exist` / setup_db connects but to the wrong database** — you
  likely already have a *native* (non-Docker) PostgreSQL running on port 5432, and it's
  intercepting the connection meant for the container. That's why `docker-compose.yml`
  maps the container to host port **5433**, not 5432 - if you changed that mapping back,
  either revert it or point `DATABASE_URL`/`READONLY_DATABASE_URL` at whatever port you
  chose instead. Check with `lsof -nP -iTCP:5432 -sTCP:LISTEN` if unsure what's already
  bound to 5432 on your machine.
- **`RuntimeError` about workers at API startup** — you (or your shell) set
  `WEB_CONCURRENCY`/`WORKERS`/`--workers N>1`. This app must run single-worker — see
  §2.5. Remove the multi-worker setting.
- **NL2SQL keeps returning "could not safely answer"** — check the API logs for the
  guard's rejection reason; a local model occasionally drifts into multi-statement or
  non-SELECT output. Try a larger model, or rephrase the question more simply.
- **`psycopg` install fails** — this project pins `psycopg[binary]` (v3) specifically
  because it has broader pre-built wheel support than `psycopg2-binary` on newer Python
  versions; make sure you're not on an old requirements.txt.
- **Embedding model download is slow/fails** — `sentence-transformers` downloads
  `all-MiniLM-L6-v2` (~80MB) from Hugging Face on first run; cached after that.
- **Agent gives odd/rambling answers** — smaller local models sometimes drift from the
  strict ReAct format. Try a larger model, or lower `temperature` in `src/core/agent.py`.
- **`ModuleNotFoundError: No module named 'src'` / `'config'`** — run commands from the
  project root using the `-m` forms shown above, not from inside a subfolder.

---

## 9. Make It Your Own (the intern assignment)

1. Edit `JARVIS_SYSTEM_PROMPT` in `config/settings.py` — change the voice, the name, the
   address ("sir" → whatever fits your character), the tone.
2. Swap or add files in `data/documents/` with your own knowledge — add a matching line
   to `INGESTION_PLAN` in `src/core/chunking.py` if you introduce a new format or
   structure that needs its own chunking function.
3. Add a new structured table to `src/db/models.py` (+ seed rows in `seed_data.py`) if
   your character has other structured operational data — no NL2SQL code changes needed,
   `schema_introspection.py` picks up new tables automatically.
4. Add or change tools in `src/core/tools.py` to match your character's own "mission" —
   the registry pattern (name, function, description, risk level) extends without
   touching `src/core/agent.py` at all.
5. Re-run `python -m scripts.ingest` (and `seed_db.py` if you changed structured data),
   restart the API, and you're live.