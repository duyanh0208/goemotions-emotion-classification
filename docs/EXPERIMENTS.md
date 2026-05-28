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
| Date | TBD |
| Config | `configs/bert_base.yaml` |
| Model | `bert-base-uncased` |
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

> TBD — Fill after running `python -m src.train --config configs/bert_base.yaml`.
> Expected baseline: F1-macro ≈ 0.46 (Demszky et al., 2020).

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
| BERT-base | Fine-tune | TBD | TBD | TBD | Full test (5427) |
| RoBERTa-base | Fine-tune | TBD | TBD | TBD | Full test (5427) |
| Gemini Flash | Zero-shot | TBD | TBD | TBD | 2K subset |
| Gemini Flash | Few-shot (k=5) | TBD | TBD | TBD | 2K subset |
| *Paper baseline* | *BERT Fine-tune* | *0.46* | *—* | *—* | *Demszky 2020* |
