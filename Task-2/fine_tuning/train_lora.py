"""
Act 1 — Fine-Tuning: bake the custom cake.

QLoRA fine-tune of a small instruction-tuned base model with Unsloth, on the
bakery customer-support dataset in data/sample_train.jsonl.

Run:
    python train_lora.py

Produces:
    fine_tuning/outputs/lora-adapter/   -- the trained LoRA adapter (small, a few MB)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def load_dataset_as_list(path):
    """Read the chat-format JSONL dataset into a plain Python list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    print(f"[1/5] Loading base model: {cfg.MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.MODEL_NAME,
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        dtype=cfg.DTYPE,
        load_in_4bit=cfg.LOAD_IN_4BIT,
    )

    if cfg.CHAT_TEMPLATE_OVERRIDE:
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(tokenizer, chat_template=cfg.CHAT_TEMPLATE_OVERRIDE)

    print("[2/5] Wrapping model with LoRA adapters (PEFT)")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.LORA_R,
        target_modules=cfg.LORA_TARGET_MODULES,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=cfg.USE_GRADIENT_CHECKPOINTING,
        random_state=cfg.SEED,
    )

    print(f"[3/5] Loading + formatting dataset: {cfg.TRAIN_DATA_PATH}")
    raw_records = load_dataset_as_list(cfg.TRAIN_DATA_PATH)

    def format_example(example):
        # Always use the tokenizer's own chat template -- never hand-roll the
        # prompt format (Slide 15, subtopic 1: "the mold").
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list(raw_records).map(format_example)
    print(f"    -> {len(dataset)} training examples formatted with the model's chat template")
    print("    -> sample formatted example:\n", dataset[0]["text"][:400], "...\n")

    print("[4/5] Configuring the trainer")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        args=SFTConfig(
            per_device_train_batch_size=cfg.PER_DEVICE_BATCH_SIZE,
            gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=cfg.WARMUP_STEPS,
            num_train_epochs=cfg.NUM_TRAIN_EPOCHS,
            learning_rate=cfg.LEARNING_RATE,
            logging_steps=cfg.LOGGING_STEPS,
            weight_decay=cfg.WEIGHT_DECAY,
            lr_scheduler_type=cfg.LR_SCHEDULER_TYPE,
            seed=cfg.SEED,
            output_dir=os.path.join(cfg.OUTPUT_DIR, "checkpoints"),
            report_to="none",
        ),
    )

    print("[5/5] Training (this is the 30-minute lab step in a live GPU session)")
    trainer_stats = trainer.train()
    print("Training complete. Stats:", trainer_stats)

    os.makedirs(cfg.LORA_ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(cfg.LORA_ADAPTER_DIR)
    tokenizer.save_pretrained(cfg.LORA_ADAPTER_DIR)
    print(f"Saved LoRA adapter to {cfg.LORA_ADAPTER_DIR}")
    print("Next: run `python inference_compare.py` for a before/after comparison.")


if __name__ == "__main__":
    main()
