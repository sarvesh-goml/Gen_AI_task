# AI Shopping Assistant

A RAG-powered single-agent AI Shopping Assistant built with FastAPI, PostgreSQL + pgvector, and OpenAI.

## Features

- **Hybrid Product Search**: Combines pgvector semantic search with structured SQL filtering (category, brand, price constraints).
- **Single Tool-Using Agent**: Built with OpenAI tool calling, maintaining conversation memory and shopping context.
- **Risk-Based Guardrails**: Low-risk actions execute automatically; medium and high-risk actions (checkout, placing orders) require explicit approval.
- **Observability**: Interception logs tracking query latency, tokens used, and calculated OpenAI costs.
- **Evaluation Framework**: Evaluating relevance, constraint satisfaction, and groundedness of recommendations.

## Running the Application

### 1. Locally

Ensure you have PostgreSQL with the `pgvector` extension running locally.

1. **Set Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/shop
   OPENAI_API_KEY=your-openai-api-key
   OPENAI_MODEL_NAME=gpt-4o-mini
   ```

2. **Install Dependencies**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run Database Seed & RAG Ingestion**:
   ```bash
   PYTHONPATH=. python3 scripts/seed.py
   ```

4. **Start the API Server**:
   ```bash
   PYTHONPATH=. uvicorn src.main:app --reload
   ```

5. **Run Evaluation**:
   ```bash
   PYTHONPATH=. python3 scripts/evaluate.py
   ```

### 2. With Docker Compose

1. **Run**:
   ```bash
   OPENAI_API_KEY=your-openai-api-key docker-compose up --build
   ```
2. **Seed & Index Products**:
   Send a POST request to `/api/seed` to populate the pgvector database:
   ```bash
   curl -X POST http://localhost:8000/api/seed
   ```

## Running Tests

```bash
PYTHONPATH=. pytest
```
