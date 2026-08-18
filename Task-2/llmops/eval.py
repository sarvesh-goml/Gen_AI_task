"""
Act 3 — LLMOps: quality control on every batch.

A tiny automated eval gate in the spirit of DeepEval/UpTrain: run the model
against a fixed set of held-out prompts, score it, and exit non-zero if it
falls below a quality threshold -- this is Slide 30's "AI Evaluation Loop"
(define good, baseline, find failures, rerun on the same dataset) wired into
a CI/CD pipeline, one item on Slide 33's deployment checklist, so a bad
prompt or model change can't silently ship.

Run standalone:
    python eval.py

Run as a CI gate (exits 1 on failure, so `&&`/CI steps stop the pipeline):
    python eval.py --threshold 0.25
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fine_tuning"))
import config as ft_cfg  # noqa: E402


def load_eval_prompts(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def rouge_l(prediction, reference):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, prediction)["rougeL"].fmeasure


def main():
    parser = argparse.ArgumentParser(description="Automated eval gate for the bakery assistant model.")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", ft_cfg.LORA_ADAPTER_DIR))
    parser.add_argument("--threshold", type=float, default=0.20,
                         help="Minimum average ROUGE-L to pass (fails the gate below this).")
    parser.add_argument("--max-latency", type=float, default=15.0,
                         help="Maximum acceptable average latency per response, in seconds.")
    args = parser.parse_args()

    from unsloth import FastLanguageModel

    print(f"Loading model for eval: {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=ft_cfg.MAX_SEQ_LENGTH,
        dtype=ft_cfg.DTYPE,
        load_in_4bit=ft_cfg.LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)

    eval_prompts = load_eval_prompts(ft_cfg.EVAL_PROMPTS_PATH)
    scores, latencies = [], []

    for ex in eval_prompts:
        messages = [{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["prompt"]}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        start = time.time()
        outputs = model.generate(
            input_ids=inputs, max_new_tokens=120, temperature=0.7,
            do_sample=True, pad_token_id=tokenizer.eos_token_id,
        )
        latency = time.time() - start
        latencies.append(latency)

        generated = outputs[0][inputs.shape[-1]:]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
        score = rouge_l(answer, ex["reference"])
        scores.append(score)

        print(f"  [{score:.3f} | {latency:.2f}s] {ex['prompt'][:60]}...")

    avg_score = sum(scores) / len(scores)
    avg_latency = sum(latencies) / len(latencies)

    print(f"\nAverage ROUGE-L: {avg_score:.3f}  (threshold: {args.threshold})")
    print(f"Average latency: {avg_latency:.2f}s  (max allowed: {args.max_latency}s)")

    passed = avg_score >= args.threshold and avg_latency <= args.max_latency
    result = {
        "model_path": args.model_path,
        "avg_rouge_l": round(avg_score, 3),
        "avg_latency_s": round(avg_latency, 3),
        "threshold": args.threshold,
        "max_latency": args.max_latency,
        "passed": passed,
    }
    print(f"\nEVAL GATE: {'PASS' if passed else 'FAIL'}")
    print(json.dumps(result, indent=2))

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/eval_gate_result.json", "w") as f:
        json.dump(result, f, indent=2)

    if not passed:
        sys.exit(1)  # non-zero exit -- this is what makes it usable as a CI gate


if __name__ == "__main__":
    main()
