# Experiment Log

Records every experiment run. Fill in each entry after a run completes.

---

## How to Fill This File

1. Copy the template block below into a new section.
2. Replace `EXP-XX` with the next sequential number.
3. Fill in Date, Config, Hardware, and Training time after the run.
4. Paste metrics from `results/metrics/<experiment_name>_metrics.json`.
5. Write a brief Observations paragraph.
6. Commit this file together with the metrics JSON.

**Template:**
```markdown
## EXP-XX: <experiment_name>

| Field | Value |
|-------|-------|
| Date | TBD |
| Config | `configs/<name>.yaml` |
| Hardware | GPU model, VRAM |
| Training time | TBD |

### Results

| Metric | Value |
|--------|-------|
| F1-macro | TBD |
| F1-micro | TBD |
| Hamming Loss | TBD |
| Best val epoch | TBD |

### Observations

> TBD
```

---

## EXP-01: bert_base_baseline

| Field | Value |
|-------|-------|
| Date | 2026-05-28 |
| Config | `configs/bert_base.yaml` |
| Model | `bert-base-uncased` |
| Hardware | NVIDIA RTX 2000 Ada Generation (17.2 GB VRAM) |
| Training time | 37m 8s |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Seed | 42 |

### Results

| Metric | Epoch 1 Val | Epoch 2 Val | Epoch 3 Val | Test |
|--------|------------|------------|------------|------|
| F1-macro | 0.4012 | 0.4137 | **0.4305** | **0.4159** |
| F1-micro | 0.4150 | 0.4459 | 0.4717 | 0.4650 |
| F1-weighted | 0.5010 | 0.5276 | 0.5359 | 0.5329 |
| Hamming Loss | 0.0992 | 0.0870 | 0.0766 | 0.0778 |
| Train Loss | 0.7009 | 0.4419 | 0.3408 | — |

Best epoch: **3** (Val F1-macro = 0.4305)

### Per-class F1 (Test)

**Top 5:**
| Emotion | F1 |
|---------|-----|
| gratitude | 0.836 |
| amusement | 0.803 |
| love | 0.750 |
| neutral | 0.682 |
| admiration | 0.627 |

**Bottom 5:**
| Emotion | F1 |
|---------|-----|
| realization | 0.190 |
| relief | 0.193 |
| disappointment | 0.220 |
| grief | 0.222 |
| nervousness | 0.246 |

### Observations

- Test F1-macro = **0.4159**, đạt ~90% paper baseline (0.46). Chênh lệch có thể do paper dùng thêm hyperparameter tuning và nhiều epochs hơn.
- Loss vẫn đang giảm ở epoch 3 (0.341) trong khi val loss bắt đầu tăng nhẹ (0.510) — dấu hiệu slight overfitting. Có thể thử epoch 4-5 với early stopping.
- Rare classes (relief, realization, grief) F1 thấp như dự đoán dù đã dùng pos_weight — đây là challenge cốt lõi của task.
- Common emotions (gratitude, amusement, love) đạt F1 > 0.75 — model học tốt với data nhiều.

---

## EXP-02: roberta_base_baseline

| Field | Value |
|-------|-------|
| Date | TBD |
| Config | `configs/roberta_base.yaml` |
| Model | `roberta-base` |
| Hardware | TBD |
| Training time | TBD |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Seed | 42 |

### Results

| Metric | Val | Test |
|--------|-----|------|
| F1-macro | TBD | TBD |
| F1-micro | TBD | TBD |
| F1-weighted | TBD | TBD |
| Hamming Loss | TBD | TBD |
| Best epoch | TBD | — |

### Observations

> TBD — Fill after running `python -m src.train --config configs/roberta_base.yaml`.
> Hypothesis: RoBERTa should outperform BERT by ~2–4 F1-macro points (improved
> pre-training: no NSP, more data, dynamic masking).

---

## EXP-03: gemini_zeroshot

| Field | Value |
|-------|-------|
| Date | TBD |
| Config | `configs/gemini_zeroshot.yaml` |
| Model | `gemini-2.0-flash` |
| Hardware | API — no local GPU |
| Inference time | TBD (~9 min for 2000 samples at 15 RPM) |
| Temperature | 0.0 |
| n_samples | 2000 (random subset, seed=42) |

### Results

| Metric | Value |
|--------|-------|
| F1-macro | TBD |
| F1-micro | TBD |
| F1-weighted | TBD |
| Hamming Loss | TBD |

### Observations

> TBD — Fill after running:
> `python -m src.llm_inference --config configs/gemini_zeroshot.yaml --n_samples 2000`
>
> Note: Checkpoint resume is supported — if interrupted, re-run the same command.
>
> Hypothesis: Zero-shot LLM will underperform fine-tuned BERT on F1-macro overall,
> but may show stronger recall on rare classes (grief, pride, relief) where the
> fine-tuned model has few training examples.

---

## EXP-04: gemini_fewshot_k5

| Field | Value |
|-------|-------|
| Date | TBD |
| Config | `configs/gemini_fewshot.yaml` |
| Model | `gemini-2.0-flash` |
| Hardware | API — no local GPU |
| Inference time | TBD |
| Temperature | 0.0 |
| n_samples | 2000 (same random subset as EXP-03) |
| k examples | 5 (fixed, curated) |

### Results

| Metric | Value |
|--------|-------|
| F1-macro | TBD |
| F1-micro | TBD |
| F1-weighted | TBD |
| Hamming Loss | TBD |

### Observations

> TBD — Fill after running:
> `python -m src.llm_inference --config configs/gemini_fewshot.yaml --n_samples 2000`
>
> Hypothesis: Few-shot (k=5) will improve over zero-shot by anchoring the output
> format and demonstrating multi-label behaviour; gains expected on rare classes
> and neutral-vs-low-emotion edge cases.

---

## Summary Comparison

> Run `python scripts/compare_results.py` to auto-generate this table and save to
> `results/metrics/comparison_table.csv`.

| Model | Method | F1-macro | F1-micro | Hamming Loss | Eval set |
|-------|--------|----------|----------|--------------|----------|
| BERT-base | Fine-tune | **0.4159** | **0.4650** | **0.0778** | Full test (5427) |
| RoBERTa-base | Fine-tune | TBD | TBD | TBD | Full test (5427) |
| Gemini 2.5 Flash | Zero-shot | TBD | TBD | TBD | 2K subset |
| Gemini 2.5 Flash | Few-shot (k=5) | TBD | TBD | TBD | 2K subset |
| *Paper baseline* | *BERT Fine-tune* | *0.46* | *—* | *—* | *Demszky 2020* |
