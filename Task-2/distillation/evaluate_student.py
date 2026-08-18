"""
Act 2 — Distillation: taste-test the cupcake.

Compares the distilled student model against the fine-tuned teacher model on
the same held-out prompts used in fine_tuning/inference_compare.py, and
reports both a quality signal (ROUGE-L vs. the reference answers) and a
practical size/speed signal (parameter count, rough latency).

Run:
    python evaluate_student.py
"""

import importlib.util
import json
import os
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cfg = _load_module("distillation_config", os.path.join(THIS_DIR, "config.py"))
ft_cfg = _load_module("fine_tuning_config", os.path.join(THIS_DIR, "..", "fine_tuning", "config.py"))


def load_eval_prompts(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def generate(model, tokenizer, system, prompt, max_new_tokens=120):
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    start = time.time()
    outputs = model.generate(
        input_ids=inputs, max_new_tokens=max_new_tokens, temperature=0.7,
        do_sample=True, pad_token_id=tokenizer.eos_token_id,
    )
    latency = time.time() - start

    generated = outputs[0][inputs.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, latency


def rouge_l(prediction, reference):
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return round(scorer.score(reference, prediction)["rougeL"].fmeasure, 3)
    except ImportError:
        return None


def count_trainable_and_total_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main():
    from unsloth import FastLanguageModel

    eval_prompts = load_eval_prompts(ft_cfg.EVAL_PROMPTS_PATH)

    print(f"Loading TEACHER: {ft_cfg.LORA_ADAPTER_DIR}")
    teacher, teacher_tok = FastLanguageModel.from_pretrained(
        model_name=ft_cfg.LORA_ADAPTER_DIR,
        max_seq_length=ft_cfg.MAX_SEQ_LENGTH,
        dtype=ft_cfg.DTYPE,
        load_in_4bit=ft_cfg.LOAD_IN_4BIT,
    )

    print(f"Loading STUDENT: {cfg.STUDENT_OUTPUT_DIR}")
    if not os.path.isdir(cfg.STUDENT_OUTPUT_DIR):
        raise FileNotFoundError(
            f"No distilled student found at {cfg.STUDENT_OUTPUT_DIR} -- run distill.py first."
        )
    student, student_tok = FastLanguageModel.from_pretrained(
        model_name=cfg.STUDENT_OUTPUT_DIR,
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        dtype=cfg.DTYPE,
        load_in_4bit=cfg.LOAD_IN_4BIT,
    )

    _, teacher_total = count_trainable_and_total_params(teacher)
    _, student_total = count_trainable_and_total_params(student)

    results = []
    for i, ex in enumerate(eval_prompts, 1):
        print(f"\n[{i}/{len(eval_prompts)}] {ex['prompt']}")

        teacher_answer, teacher_latency = generate(teacher, teacher_tok, ex["system"], ex["prompt"])
        student_answer, student_latency = generate(student, student_tok, ex["system"], ex["prompt"])

        t_score = rouge_l(teacher_answer, ex["reference"])
        s_score = rouge_l(student_answer, ex["reference"])

        print(f"  TEACHER (ROUGE-L={t_score}, {teacher_latency:.2f}s): {teacher_answer}")
        print(f"  STUDENT (ROUGE-L={s_score}, {student_latency:.2f}s): {student_answer}")

        results.append({
            "prompt": ex["prompt"],
            "teacher_answer": teacher_answer, "teacher_rouge_l": t_score, "teacher_latency_s": round(teacher_latency, 3),
            "student_answer": student_answer, "student_rouge_l": s_score, "student_latency_s": round(student_latency, 3),
        })

    avg_t = sum(r["teacher_rouge_l"] or 0 for r in results) / len(results)
    avg_s = sum(r["student_rouge_l"] or 0 for r in results) / len(results)
    avg_t_lat = sum(r["teacher_latency_s"] for r in results) / len(results)
    avg_s_lat = sum(r["student_latency_s"] for r in results) / len(results)

    summary = {
        "teacher_total_params": teacher_total,
        "student_total_params": student_total,
        "param_reduction_pct": round(100 * (1 - student_total / teacher_total), 1),
        "avg_teacher_rouge_l": round(avg_t, 3),
        "avg_student_rouge_l": round(avg_s, 3),
        "quality_gap": round(avg_t - avg_s, 3),
        "avg_teacher_latency_s": round(avg_t_lat, 3),
        "avg_student_latency_s": round(avg_s_lat, 3),
        "speedup_x": round(avg_t_lat / avg_s_lat, 2) if avg_s_lat > 0 else None,
    }

    out_path = os.path.join(THIS_DIR, "outputs", "student_vs_teacher.json")
    os.makedirs(os.path.join(THIS_DIR, "outputs"), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "examples": results}, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}\nSUMMARY: {json.dumps(summary, indent=2)}")
    print(f"\nSaved full comparison to {out_path}")
    print("This is the trade-off from Slide 21: quality gap vs. speed/size gain --")
    print("decide per-project whether the cupcake is good enough for the occasion.")


if __name__ == "__main__":
    main()
