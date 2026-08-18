# Act 2 — Distillation ("bake the cupcake")

Trains a smaller **student** model to imitate the **teacher** (the fine-tuned model from
`../fine_tuning/`), using soft labels — the teacher's full output distribution — rather than
just its final generated text.

## Files

| File | What it does |
|---|---|
| `config.py` | Student model choice, LoRA settings, and the two distillation knobs: `TEMPERATURE` and `ALPHA`. |
| `distill.py` | Loads the frozen teacher and a trainable student, runs a compact training loop combining KL-divergence (match the teacher) with normal task loss (stay correct), and saves the student. |
| `evaluate_student.py` | Compares student vs. teacher on the same held-out prompts: ROUGE-L quality gap, parameter count reduction, and rough latency. |

## Run

```bash
cd ../fine_tuning && python train_lora.py && cd ../distillation   # need a teacher first
python distill.py
python evaluate_student.py
```

## How the loss works

```
loss = ALPHA * KL(student_soft || teacher_soft) + (1 - ALPHA) * cross_entropy(student, true_tokens)
```

- **Temperature** softens both distributions before the KL term — a higher temperature spreads
  probability mass over more tokens, exposing more of "what the teacher considered as
  runners-up," which is the whole point of using soft labels over hard labels (Slide 21).
- **Alpha** controls how much the student should imitate the teacher's *style of uncertainty*
  versus just getting the literal next token right. `ALPHA = 0.7` in `config.py` leans toward
  imitation; drop it toward `0.3` if the student starts sounding vague rather than confident.

## Reading the results

`evaluate_student.py` writes `outputs/student_vs_teacher.json` with a `summary` block like:

```json
{
  "teacher_total_params": 1235814400,
  "student_total_params": 1235814400,
  "param_reduction_pct": 0.0,
  "avg_teacher_rouge_l": 0.41,
  "avg_student_rouge_l": 0.33,
  "quality_gap": 0.08,
  "speedup_x": 1.4
}
```

(Because this demo keeps the same small base architecture for portability, the headline
parameter-count reduction comes from `STUDENT_LORA_R` being smaller, not a different base
model — swap `STUDENT_MODEL_NAME` in `config.py` for a genuinely smaller architecture to see a
bigger size/speed win, exactly as you're encouraged to do for your own EOD assignment.)

The number that matters is the **trade-off**, not either score alone: how much quality did you
give up, for how much speed or cost saved — and whether that trade makes sense for your task.
