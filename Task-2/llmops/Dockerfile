# Wanderlust Travels — Travel Assistant API
#
# Slide 27's "food truck" default: a portable container that wraps the
# fine-tuned model behind a FastAPI endpoint. Build from the repo root so the
# image can also pull in ../fine_tuning/outputs (the trained adapter).
#
# Build (from the repo root, not this folder):
#   docker build -f llmops/Dockerfile -t travel-assistant-api .
# Run:
#   docker run -p 8000:8000 -e MODEL_PATH=/app/fine_tuning/outputs/lora-adapter travel-assistant-api

FROM python:3.11-slim AS base

# System deps needed by torch / bitsandbytes / git-installations at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached across code-only changes
COPY requirements.txt ./requirements.txt
COPY llmops/requirements.txt ./llmops-requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r llmops-requirements.txt

# Now copy the actual application code and the trained model artifacts
COPY fine_tuning/ ./fine_tuning/
COPY llmops/ ./llmops/

# Configuration via environment variables, never baked into the image itself
ENV MODEL_PATH=/app/fine_tuning/outputs/lora-adapter
ENV MAX_NEW_TOKENS=150
ENV LOAD_IN_4BIT=true

WORKDIR /app/llmops
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
