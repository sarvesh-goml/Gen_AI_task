"""
Central configuration for the fine-tuning step (Act 1 — "bake the custom cake").

Everything a workshop attendee is likely to want to change lives in this one file:
the base model, LoRA/QLoRA settings, dataset paths, and training hyperparameters.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)

TRAIN_DATA_PATH = os.path.join(REPO_ROOT, "data", "sample_train.jsonl")
EVAL_PROMPTS_PATH = os.path.join(REPO_ROOT, "data", "eval_prompts.jsonl")

OUTPUT_DIR = os.path.join(THIS_DIR, "outputs")
LORA_ADAPTER_DIR = os.path.join(OUTPUT_DIR, "lora-adapter")
MERGED_MODEL_DIR = os.path.join(OUTPUT_DIR, "merged-model")

# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------
# A small instruction-tuned model keeps this runnable on a single consumer GPU
# (or even CPU, slowly) — swap for a larger unsloth/* model if you have more VRAM.
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024
DTYPE = None          # None = auto-detect (bfloat16 on Ampere+, float16 otherwise)
LOAD_IN_4BIT = True    # QLoRA. Set False if bitsandbytes/4-bit isn't available on your machine.

# ---------------------------------------------------------------------------
# LoRA (PEFT) settings — see Slide 12 of the deck for what each knob means
# ---------------------------------------------------------------------------
LORA_R = 16                 # rank: capacity vs. cost knob. 8-16 is a good starting default.
LORA_ALPHA = 16
LORA_DROPOUT = 0.0          # Unsloth is optimized for dropout = 0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
USE_GRADIENT_CHECKPOINTING = "unsloth"   # Unsloth's memory-efficient checkpointing

# ---------------------------------------------------------------------------
# Training hyperparameters — the "oven temperature and time"
# ---------------------------------------------------------------------------
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
WARMUP_STEPS = 5
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "linear"
SEED = 3407
LOGGING_STEPS = 1

# ---------------------------------------------------------------------------
# Chat template — always defer to the tokenizer's own template (Slide 15,
# subtopic 1: "wrong mold, right batter, broken shape"). Set here only if you
# need to force a specific template name recognised by Unsloth's
# `get_chat_template` helper; leave as None to use the model's built-in one.
# ---------------------------------------------------------------------------
CHAT_TEMPLATE_OVERRIDE = None   # e.g. "llama-3.1" — see unsloth.chat_templates
