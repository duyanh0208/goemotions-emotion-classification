# 🧪 Experiments Log

Log chi tiết các experiments đã chạy. Update sau mỗi run.

---

## Template entry

```
## Experiment: <name>
**Date:** YYYY-MM-DD
**Config:** configs/<name>.yaml
**W&B Run:** <link>

### Setup
- Hardware: <GPU>
- Training time: <duration>

### Results
| Metric | Val | Test |
|--------|-----|------|
| F1-macro | - | - |
| F1-micro | - | - |
| Hamming Loss | - | - |

### Observations
- ...

### Issues
- ...
```

---

## Experiment: bert_base_baseline

**Date:** TBD
**Config:** `configs/bert_base.yaml`
**W&B Run:** TBD

### Setup
- Hardware: RTX 2000 Ada (16GB)
- Training time: TBD

### Results
| Metric | Val | Test |
|--------|-----|------|
| F1-macro | - | - |
| F1-micro | - | - |
| Hamming Loss | - | - |

### Observations
- TBD

---

## Experiment: roberta_base_baseline

**Date:** TBD
**Config:** `configs/roberta_base.yaml`

(Pending)

---

## Experiment: gemini_zeroshot

**Date:** TBD
**Config:** `configs/gemini_zeroshot.yaml`

(Pending)

---

## Experiment: gemini_fewshot

**Date:** TBD
**Config:** `configs/gemini_fewshot.yaml`

(Pending)

---

## Summary Comparison

| Experiment | F1-macro | F1-micro | Notes |
|-----------|----------|----------|-------|
| BERT-base | - | - | Baseline |
| RoBERTa-base | - | - | Comparison |
| Gemini zero-shot | - | - | No training |
| Gemini few-shot (k=5) | - | - | In-context |
