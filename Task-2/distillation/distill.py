"""
Act 2 — Distillation: bake the cupcake version.

Teaches a small "student" model to imitate the fine-tuned "teacher" model from
Act 1, using soft labels (the teacher's full output distribution) rather than
just its final generated text ("hard labels").

Loss = alpha * KL-divergence(student || teacher) + (1 - alpha) * task loss
(standard cross-entropy against the training text itself)

This is a compact, readable reference implementation -- not a
production-scale distillation trainer. It is meant to be stepped through
live, not stared at as a black box.

Run:
    python distill.py
"""

import importlib.util
import json
import os
import sys

import torch
import torch.nn.functional as F

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(module_name, file_path):
    """Load a config.py file under a unique module name so it doesn't collide
    with the sibling fine_tuning/config.py, which also imports as `config`."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cfg = _load_module("distillation_config", os.path.join(THIS_DIR, "config.py"))
ft_cfg = _load_module("fine_tuning_config", os.path.join(THIS_DIR, "..", "fine_tuning", "config.py"))


def load_dataset_as_list(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """
    student_logits, teacher_logits: [batch, seq_len, vocab]
    labels: [batch, seq_len] with -100 for positions to ignore (e.g. prompt tokens)
    """
    # --- soft-label term: KL divergence between softened student & teacher distributions ---
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

    # --- hard-label / task term: normal next-token cross-entropy against the real text ---
    task_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )

    return alpha * kd_loss + (1 - alpha) * task_loss, kd_loss.item(), task_loss.item()


def main():
    from unsloth import FastLanguageModel

    print(f"[1/5] Loading TEACHER (fine-tuned model): {ft_cfg.LORA_ADAPTER_DIR}")
    if not os.path.isdir(ft_cfg.LORA_ADAPTER_DIR):
        raise FileNotFoundError(
            f"No fine-tuned teacher found at {ft_cfg.LORA_ADAPTER_DIR} -- "
            "run fine_tuning/train_lora.py first."
        )
    teacher, teacher_tok = FastLanguageModel.from_pretrained(
        model_name=ft_cfg.LORA_ADAPTER_DIR,
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        dtype=ft_cfg.DTYPE,
        load_in_4bit=ft_cfg.LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(teacher)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print(f"[2/5] Loading STUDENT (smaller base model): {cfg.STUDENT_MODEL_NAME}")
    student, student_tok = FastLanguageModel.from_pretrained(
        model_name=cfg.STUDENT_MODEL_NAME,
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        dtype=cfg.DTYPE,
        load_in_4bit=cfg.LOAD_IN_4BIT,
    )
    student = FastLanguageModel.get_peft_model(
        student,
        r=cfg.STUDENT_LORA_R,
        target_modules=cfg.STUDENT_LORA_TARGET_MODULES,
        lora_alpha=cfg.STUDENT_LORA_R,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.SEED,
    )
    FastLanguageModel.for_training(student)

    print(f"[3/5] Loading + formatting dataset: {ft_cfg.TRAIN_DATA_PATH}")
    raw_records = load_dataset_as_list(ft_cfg.TRAIN_DATA_PATH)

    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.LEARNING_RATE)

    print(f"[4/5] Distilling for {cfg.NUM_EPOCHS} epoch(s) "
          f"(temperature={cfg.TEMPERATURE}, alpha={cfg.ALPHA})")
    student.train()
    step = 0
    for epoch in range(cfg.NUM_EPOCHS):
        for record in raw_records:
            text = student_tok.apply_chat_template(
                record["messages"], tokenize=False, add_generation_prompt=False
            )
            enc = student_tok(
                text, return_tensors="pt", truncation=True, max_length=cfg.MAX_SEQ_LENGTH
            ).to(student.device)
            labels = enc["input_ids"].clone()

            with torch.no_grad():
                teacher_out = teacher(**enc.to(teacher.device))
                teacher_logits = teacher_out.logits.to(student.device)

            student_out = student(**enc)
            student_logits = student_out.logits

            # Align sequence lengths defensively (teacher/student tokenizers should match
            # here since both derive from compatible base vocabularies in this demo).
            min_len = min(student_logits.size(1), teacher_logits.size(1))
            loss, kd, task = distillation_loss(
                student_logits[:, :min_len, :],
                teacher_logits[:, :min_len, :],
                labels[:, :min_len],
                temperature=cfg.TEMPERATURE,
                alpha=cfg.ALPHA,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % cfg.LOG_EVERY == 0:
                print(f"  epoch {epoch + 1} step {step}: "
                      f"loss={loss.item():.4f} (kd={kd:.4f}, task={task:.4f})")

    print("[5/5] Saving distilled student model")
    os.makedirs(cfg.STUDENT_OUTPUT_DIR, exist_ok=True)
    student.save_pretrained(cfg.STUDENT_OUTPUT_DIR)
    student_tok.save_pretrained(cfg.STUDENT_OUTPUT_DIR)
    print(f"Saved distilled student to {cfg.STUDENT_OUTPUT_DIR}")
    print("Next: run `python evaluate_student.py` to compare student vs. teacher.")


if __name__ == "__main__":
    main()
