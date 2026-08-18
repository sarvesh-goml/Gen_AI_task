"""
Configuration for the distillation step (Act 2 — "bake the cupcake").

The teacher is the fine-tuned model produced by ../fine_tuning/train_lora.py
(see fine_tuning/config.py for its path). The student is a smaller model that
learns to imitate it.
"""

import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENT_OUTPUT_DIR = os.path.join(THIS_DIR, "outputs", "student-model")

# ---------------------------------------------------------------------------
# Student model — deliberately smaller/faster than the teacher's base model.
# For a real project pick a genuinely smaller architecture; here we keep the
# same small family for portability across laptops, but shrink LoRA rank and
# sequence length so the student is cheaper to run at inference time.
# ---------------------------------------------------------------------------
STUDENT_MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 512
DTYPE = None
LOAD_IN_4BIT = True

STUDENT_LORA_R = 8   # smaller than the teacher's r=16 -- less capacity, cheaper student
STUDENT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ---------------------------------------------------------------------------
# Distillation loss settings
# ---------------------------------------------------------------------------
TEMPERATURE = 2.0   # softens both distributions before computing KL-divergence
ALPHA = 0.7          # weight on the soft-label (KD) term vs. the hard-label task term
NUM_EPOCHS = 2
LEARNING_RATE = 1e-4
LOG_EVERY = 5
SEED = 3407
