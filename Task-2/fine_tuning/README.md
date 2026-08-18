# Act 1 — Fine-Tuning ("bake the custom cake")

QLoRA fine-tune of `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` (see `config.py` to swap models)
on the bakery customer-support dataset in `../data/sample_train.jsonl`, using
[Unsloth](https://github.com/unslothai/unsloth) for fast, memory-efficient LoRA training.

## Files

| File | What it does |
|---|---|
| `config.py` | Every hyperparameter in one place: base model, LoRA rank/target modules, training args. |
| `train_lora.py` | Loads the base model in 4-bit, wraps it with a LoRA adapter, formats the dataset with the tokenizer's own chat template, and trains. Saves the adapter to `outputs/lora-adapter/`. |
| `inference_compare.py` | Loads both the base model and the fine-tuned model, runs both on the 5 held-out prompts in `../data/eval_prompts.jsonl`, prints them side by side, and scores each with ROUGE-L. |

## Run

```bash
python train_lora.py
python inference_compare.py
```

## Key design choices (and why)

- **QLoRA, not full fine-tuning.** The base model loads in 4-bit (`LOAD_IN_4BIT = True`); only
  the small LoRA matrices are trained. This is what lets the whole thing run on a single
  consumer GPU instead of a cluster.
- **`tokenizer.apply_chat_template()` is used everywhere**, never a hand-written prompt string.
  This is the single most common way fine-tunes silently break — see Slide 15 of the deck.
- **`r=16`, `lora_alpha=16`** in `config.py` is a solid starting default. If you want to see the
  rank/capacity trade-off live, try `r=4` (underfits faster, trains faster) vs `r=64` (more
  capacity, more memory) and compare the loss curves.
- **Evaluation pairs a metric with your own eyes.** `inference_compare.py` prints ROUGE-L next
  to every answer, but the point of printing the full text is that you read it — a higher ROUGE-L
  score does not always mean a better answer.

## Expected output

After training, `outputs/lora-adapter/` contains the adapter weights (a few megabytes, not a
full model copy). After `inference_compare.py`, `outputs/before_after_comparison.json` has the
full set of base vs. fine-tuned answers plus scores, ready to paste into a model card or your
assignment write-up.
