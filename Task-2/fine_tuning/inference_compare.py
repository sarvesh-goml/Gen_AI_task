"""
Act 1 — Fine-Tuning: before/after comparison.

Loads the base model and the fine-tuned (base + LoRA adapter) model, runs both
on the held-out prompts in data/eval_prompts.jsonl, and prints a side-by-side
comparison plus a quick ROUGE-L score against the reference answers.

This mirrors Slide 15's "Evaluation" subtopic: pair an automated metric
(ROUGE-L here) with your own human judgment of the printed outputs -- don't
trust the metric alone.

Run:
    python inference_compare.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


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

    FastLanguageModel.for_inference(model)  # enables Unsloth's faster inference path
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def rouge_l(prediction, reference):
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        score = scorer.score(reference, prediction)["rougeL"].fmeasure
        return round(score, 3)
    except ImportError:
        return None


def main():
    from unsloth import FastLanguageModel

    eval_prompts = load_eval_prompts(cfg.EVAL_PROMPTS_PATH)
    print(f"Loaded {len(eval_prompts)} held-out prompts from {cfg.EVAL_PROMPTS_PATH}\n")

    print("Loading BASE model (no fine-tuning) ...")
    base_model, base_tok = FastLanguageModel.from_pretrained(
        model_name=cfg.MODEL_NAME,
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        dtype=cfg.DTYPE,
        load_in_4bit=cfg.LOAD_IN_4BIT,
    )

    print("Loading FINE-TUNED model (base + LoRA adapter) ...")
    if not os.path.isdir(cfg.LORA_ADAPTER_DIR):
        raise FileNotFoundError(
            f"No adapter found at {cfg.LORA_ADAPTER_DIR} -- run train_lora.py first."
        )
    ft_model, ft_tok = FastLanguageModel.from_pretrained(
        model_name=cfg.LORA_ADAPTER_DIR,
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        dtype=cfg.DTYPE,
        load_in_4bit=cfg.LOAD_IN_4BIT,
    )

    results = []
    for i, ex in enumerate(eval_prompts, 1):
        print(f"\n{'=' * 80}\n[{i}/{len(eval_prompts)}] PROMPT: {ex['prompt']}")

        base_answer = generate(base_model, base_tok, ex["system"], ex["prompt"])
        ft_answer = generate(ft_model, ft_tok, ex["system"], ex["prompt"])

        base_score = rouge_l(base_answer, ex["reference"])
        ft_score = rouge_l(ft_answer, ex["reference"])

        print(f"\n  BASE       (ROUGE-L={base_score}): {base_answer}")
        print(f"  FINE-TUNED (ROUGE-L={ft_score}): {ft_answer}")
        print(f"  REFERENCE                   : {ex['reference']}")

        results.append({
            "prompt": ex["prompt"],
            "base_answer": base_answer,
            "fine_tuned_answer": ft_answer,
            "reference": ex["reference"],
            "base_rouge_l": base_score,
            "fine_tuned_rouge_l": ft_score,
        })

    out_path = os.path.join(cfg.OUTPUT_DIR, "before_after_comparison.json")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    avg_base = sum(r["base_rouge_l"] or 0 for r in results) / len(results)
    avg_ft = sum(r["fine_tuned_rouge_l"] or 0 for r in results) / len(results)
    print(f"\n{'=' * 80}\nAverage ROUGE-L  |  base: {avg_base:.3f}   fine-tuned: {avg_ft:.3f}")
    print(f"Saved full comparison to {out_path}")
    print("\nRemember Slide 15's lesson: don't trust ROUGE-L alone -- read the actual")
    print("answers above and judge tone, correctness, and whether it sounds like Atlas.")


if __name__ == "__main__":
    main()
