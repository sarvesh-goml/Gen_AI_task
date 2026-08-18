# Act 3 — LLMOps ("run the bakery")

Wraps the fine-tuned model in a servable, testable, containerized API — the difference between
a model that works in a notebook and one a real team can trust.

## Files

| File | What it does |
|---|---|
| `app.py` | Thin FastAPI service: `/health` and `/generate`. Config (model path, system prompt, token limits) comes entirely from environment variables — nothing is hardcoded. |
| `eval.py` | Automated eval gate, DeepEval/UpTrain-style: runs the model against the held-out prompts, scores with ROUGE-L, and **exits non-zero if it falls below a threshold** — wire this into CI so a bad change can't silently ship. |
| `Dockerfile` | Containerizes the API. Built from the **repo root** (not this folder) so it can copy in the trained model from `../fine_tuning/outputs`. |
| `requirements.txt` | Extra deps needed only for serving (FastAPI/uvicorn), kept separate from the root `requirements.txt` used by training. |

## Run locally

```bash
pip install -r requirements.txt        # adds FastAPI + uvicorn
export MODEL_PATH=../fine_tuning/outputs/lora-adapter
uvicorn app:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Do you have any gluten-free cakes?"}'
```

## Run the eval gate

```bash
python eval.py --threshold 0.20 --max-latency 15
echo $?   # 0 = passed, 1 = failed -- this is the exit code a CI step checks
```

This is the automated-QC idea from Slide 30 (the AI Evaluation Loop): don't evaluate once at launch and call it done —
run the same check on every change, and block the pipeline if quality regresses.

## Run in Docker

```bash
cd ..   # repo root
docker build -f llmops/Dockerfile -t cake-bakery-api .
docker run -p 8000:8000 cake-bakery-api
```

## A minimal CI step (GitHub Actions sketch)

```yaml
# .github/workflows/ci.yml
name: CI
on: [push]
jobs:
  eval-and-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt -r llmops/requirements.txt
      - run: python llmops/eval.py --threshold 0.20   # blocks the pipeline on regression
      - run: docker build -f llmops/Dockerfile -t cake-bakery-api .
```

This is intentionally a sketch, not a ready-to-run workflow (it assumes a trained model is
already checked in or fetched from storage) — the point is showing *where* the eval gate sits
in a real pipeline, per Slides 30 and 33.

## What's deliberately left out (and why)

To keep this a demo you can read end to end in one sitting, this repo does **not** include
semantic caching or cost optimization (Slide 32), Llama Guard-style moderation or
human-in-the-loop approval gates (Slide 29's guardrails), or LangSmith/Arize-style tracing
(Slide 28's three pillars of observability) — all concept-only for tonight, covered in Act 3 of
the deck but intentionally not part of the basic demo or assignment rubric. Adding any one of
them to this `llmops/` folder is a great way to go beyond the minimum on tonight's assignment if
you want extra credit.
