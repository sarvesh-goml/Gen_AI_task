# Unsloth Cake Demo — Fine-Tuning, Distillation & LLMOps

A small, self-contained project used as the live demo for the "Fine-Tuning, Distillation &
LLMOps" training session. It is intentionally tiny — the point is to show the **full pipeline
end to end**, not to produce a state-of-the-art model.

Sticking with the session's analogy: this repo bakes a **custom cake** (fine-tuning), bakes a
**cupcake version** of it (distillation), and then **runs a bakery** around it (LLMOps).

The task is a "bakery customer support assistant" — a small open model is fine-tuned to answer
customer questions in a warm, on-brand voice for a fictional bakery. It's a deliberately simple,
easy-to-inspect task so the *pipeline* stays the star, not the dataset.

```
unsloth-cake-demo/
├── data/
│   └── sample_train.jsonl        # ~40 instruction examples (chat format)
├── fine_tuning/                  # ACT 1 — bake the custom cake (QLoRA with Unsloth)
│   ├── config.py
│   ├── train_lora.py
│   ├── inference_compare.py
│   └── README.md
├── distillation/                 # ACT 2 — bake the cupcake (teacher → student KD)
│   ├── distill.py
│   ├── evaluate_student.py
│   └── README.md
├── llmops/                       # ACT 3 — run the bakery (serve, eval, containerize)
│   ├── app.py
│   ├── eval.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
├── requirements.txt
├── .gitignore
└── README.md                     # you are here
```

## 1. Prerequisites

- Python 3.10 or 3.11
- An NVIDIA GPU with at least 8 GB VRAM (a free Colab/Kaggle T4 works fine). CPU-only will run
  but fine-tuning will be very slow — see the "No GPU?" note below.
- CUDA-capable PyTorch. Unsloth installs a matching PyTorch build for you in most cases, but on
  an unusual environment follow [Unsloth's install guide](https://github.com/unslothai/unsloth)
  for your exact CUDA version.

## 2. Local setup

```bash
# 1. Clone your copy of this repo, then enter it
git clone <your-fork-url> unsloth-cake-demo
cd unsloth-cake-demo

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies (this installs Unsloth + torch + trl + peft + bitsandbytes, etc.)
pip install --upgrade pip
pip install -r requirements.txt

# 4. Sanity-check the install
python -c "import torch, unsloth; print('CUDA available:', torch.cuda.is_available())"
```

If `pip install unsloth` fails on your machine, use Unsloth's official install command for your
CUDA version from https://github.com/unslothai/unsloth#installation-instructions — everything
else in this repo is CUDA-version agnostic.

## 3. Run the pipeline, in order

### Act 1 — Fine-Tuning (`fine_tuning/`)

```bash
cd fine_tuning
python train_lora.py                # QLoRA fine-tune, saves adapter to ./outputs/lora-adapter
python inference_compare.py         # base vs fine-tuned, side by side, on 5 held-out prompts
cd ..
```

See `fine_tuning/README.md` for what each script does and the key hyperparameters.

### Act 2 — Distillation (`distillation/`)

```bash
cd distillation
python distill.py                   # teacher (fine-tuned model) -> student (smaller model)
python evaluate_student.py          # student vs teacher: quality gap + size/speed comparison
cd ..
```

### Act 3 — LLMOps (`llmops/`)

```bash
cd llmops
pip install -r requirements.txt     # adds fastapi + uvicorn on top of the root requirements

# Run the serving API locally
uvicorn app:app --reload --port 8000
# In another terminal:
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Do you have any gluten-free cakes?"}'

# Run the automated eval gate (mirrors what a CI step would run)
python eval.py
cd ..

# Build and run the container (build context is the repo root, not llmops/,
# so the image can also copy in the trained adapter from fine_tuning/outputs)
docker build -f llmops/Dockerfile -t cake-bakery-api .
docker run -p 8000:8000 cake-bakery-api
```

## 4. No GPU available?

Everything is written to *also* run on CPU by setting `load_in_4bit=False` and using a very small
model in `fine_tuning/config.py` (the default `MODEL_NAME` already points at a 1B-parameter
model for exactly this reason). It will be slow but will complete. For the live demo, we
recommend running the fine-tuning and distillation steps once ahead of time on a GPU (a free
Colab T4 is enough) and committing the resulting `outputs/` folders, then walking through the
code live and showing the pre-generated before/after comparisons instead of training from
scratch in front of the room.

## 5. Adapting this for your own EOD assignment

This repo is deliberately the *shape* your assignment should take, not the *content* — swap in
your own task and dataset in `data/`, adjust `fine_tuning/config.py`, and keep the same
three-folder structure (`fine_tuning/`, `distillation/`, `llmops/`) so your submission is easy
to review against the rubric. See the shared assignment tracker sheet for exactly what to fill
in before you submit your GitHub link.

## 6. Troubleshooting

| Problem | Likely fix |
|---|---|
| `ImportError: unsloth requires...` | Reinstall using the CUDA-specific command from Unsloth's README rather than plain `pip install unsloth`. |
| `CUDA out of memory` | Lower `MAX_SEQ_LENGTH` or `PER_DEVICE_BATCH_SIZE` in `fine_tuning/config.py`, or switch to a smaller `MODEL_NAME`. |
| `bitsandbytes` fails to load on Windows/macOS | 4-bit quantization needs a Linux + NVIDIA GPU environment (or WSL2 on Windows). On macOS/CPU-only, set `LOAD_IN_4BIT = False` and expect a much smaller/slower run. |
| Docker build is slow | The first build downloads the base Python image and installs PyTorch — subsequent builds are cached and much faster. |
