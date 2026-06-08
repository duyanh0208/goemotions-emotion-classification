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
| Date | 2026-05-28 |
| Config | `configs/roberta_base.yaml` |
| Model | `roberta-base` |
| Hardware | NVIDIA RTX 2000 Ada Generation (17.2 GB VRAM) |
| Training time | 37m 57s |
| Epochs | 3 |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Seed | 42 |

### Results

| Metric | Epoch 1 Val | Epoch 2 Val | Epoch 3 Val | Test |
|--------|------------|------------|------------|------|
| F1-macro | 0.4049 | 0.3785 | **0.4164** | **0.4111** |
| F1-micro | 0.4302 | 0.4216 | 0.4658 | 0.4618 |
| F1-weighted | 0.5050 | 0.5115 | 0.5302 | 0.5289 |
| Hamming Loss | 0.0914 | 0.0953 | 0.0788 | 0.0795 |
| Train Loss | 0.7088 | 0.4716 | 0.3764 | — |

Best epoch: **3** (Val F1-macro = 0.4164)

### Per-class F1 (Test)

**Top 5:**
| Emotion | F1 |
|---------|-----|
| gratitude | 0.838 |
| amusement | 0.748 |
| love | 0.732 |
| neutral | 0.684 |
| admiration | 0.630 |

**Bottom 5:**
| Emotion | F1 |
|---------|-----|
| relief | 0.184 |
| grief | 0.196 |
| disappointment | 0.210 |
| nervousness | 0.234 |
| embarrassment | 0.249 |

### Observations

- Test F1-macro = **0.4111**, thấp hơn BERT (0.4159) một chút — **trái với hypothesis** ban đầu (kỳ vọng RoBERTa tốt hơn 2-4 điểm).
- Val F1-macro epoch 2 bị drop (0.378) rồi recover epoch 3 — RoBERTa có training curve không ổn định hơn BERT trong setup này.
- Pattern per-class tương tự BERT: emotions rõ ràng cao (gratitude, amusement), rare classes thấp (relief, grief).
- Kết luận: với cùng hyperparameters 3 epochs, BERT và RoBERTa cho kết quả tương đương. RoBERTa có thể cần tune thêm lr hoặc epochs.

---

## EXP-03: llama_zeroshot

| Field | Value |
|-------|-------|
| Date | 2026-06-08 |
| Config | `configs/llama_zeroshot.yaml` |
| Model | `meta-llama/Llama-3.2-3B-Instruct` |
| Hardware | NVIDIA RTX 2000 Ada Generation (17.2 GB VRAM) |
| Inference time | 44m 18s (2000 samples, ~1.33s/sample) |
| Temperature | 0.0 (do_sample=False) |
| n_samples | 2000 (random subset, seed=42) |
| Quantization | false (float16 trên CUDA) |

### Results

| Metric | Value |
|--------|-------|
| F1-macro | **0.2126** |
| F1-micro | 0.2306 |
| F1-weighted | 0.2914 |
| Hamming Loss | 0.1068 |
| n_samples | 2000 |

### Per-class F1 (selected)

**Top 5:**
| Emotion | F1 |
|---------|-----|
| gratitude | 0.662 |
| amusement | 0.519 |
| love | 0.476 |
| admiration | 0.469 |
| joy | 0.329 |

**Bottom 5:**
| Emotion | F1 |
|---------|-----|
| nervousness | 0.034 |
| embarrassment | 0.036 |
| pride | 0.035 |
| annoyance | 0.049 |
| disgust | 0.077 |

### Observations

- F1-macro = **0.2126**, thấp hơn Qwen2.5 3B (0.2219) một chút — nhưng pattern khác hẳn.
- **Llama mạnh hơn Qwen ở rare/subtle classes:** grief=0.125 (Qwen=0.000), caring=0.214 (Qwen=0.000), neutral=0.321 (Qwen=0.266). Đây là bằng chứng partial support cho RQ3 khi so sánh giữa các LLM.
- **Hamming Loss cao hơn Qwen** (0.1068 vs 0.0713): Llama over-predict nhiều labels hơn — chiến lược aggressive hơn Qwen.
- **Trade-off:** Llama có recall cao hơn (bắt được nhiều rare emotions) nhưng precision thấp hơn; Qwen precision cao hơn nhưng bỏ sót hoàn toàn nhiều rare classes.
- Cả hai LLM đều kém xa fine-tuned BERT (0.4159) — **RQ1 confirmed rất mạnh**.

---

## EXP-04: qwen_zeroshot

| Field | Value |
|-------|-------|
| Date | 2026-06-08 |
| Config | `configs/qwen_zeroshot.yaml` |
| Model | `Qwen/Qwen2.5-3B-Instruct` |
| Hardware | NVIDIA RTX 2000 Ada Generation (17.2 GB VRAM) |
| Inference time | 15m 53s (2000 samples, ~0.48s/sample) |
| Temperature | 0.0 (do_sample=False) |
| n_samples | 2000 (random subset, seed=42) |
| Quantization | false (float16 trên CUDA) |

### Results

| Metric | Value |
|--------|-------|
| F1-macro | **0.2219** |
| F1-micro | 0.2717 |
| F1-weighted | 0.2803 |
| Hamming Loss | 0.0713 |
| n_samples | 2000 |

### Per-class F1 (selected)

**Top 5:**
| Emotion | F1 |
|---------|-----|
| gratitude | 0.692 |
| admiration | 0.470 |
| love | 0.449 |
| sadness | 0.449 |
| fear | 0.440 |

**Bottom 5:**
| Emotion | F1 |
|---------|-----|
| caring | 0.000 |
| grief | 0.000 |
| pride | 0.000 |
| approval | 0.030 |
| remorse | 0.091 |

### Observations

- F1-macro = **0.2219**, thấp hơn BERT fine-tune (0.4159) gần **2 lần** → **RQ1 confirmed mạnh**: fine-tuned model vượt LLM zero-shot đáng kể.
- **Hypothesis RQ3 bị bác bỏ:** LLM không mạnh hơn ở rare classes — grief=0.000, caring=0.000, pride=0.000, thấp hơn cả BERT (grief=0.222, caring≈0.3).
- **Neutral recall kém** (F1=0.266 vs BERT 0.682): Qwen có xu hướng over-predict emotions — thêm nhãn vào các text thực ra là neutral. Ví dụ: "KAMALA 2020!!!" → Qwen dự đoán [excitement], ground truth = [neutral].
- **Qwen mạnh hơn ở single clear-emotion cases:** Khi chỉ có 1 emotion rõ ràng (fear=0.440, gratitude=0.692), Qwen khá tốt. Nhưng với multi-label phức tạp hay cảm xúc tinh tế → fail hoàn toàn.
- **Error analysis** (2000 samples): disagreement rate với BERT = 93.1%, BERT closer to truth = 40.4%, LLM closer = 5.1%, LLM exact wins = 9.1% (181 cases), BERT exact wins = 8.0% (160 cases). LLM wins nhiều hơn BERT ở exact match nhờ precision cao, nhưng BERT thường gần đúng hơn (partial credit).

---

## Summary Comparison

> Run `python scripts/compare_results.py` to auto-generate this table and save to
> `results/metrics/comparison_table.csv`.

| Model | Method | F1-macro | F1-micro | Hamming Loss | Eval set |
|-------|--------|----------|----------|--------------|----------|
| BERT-base | Fine-tune | **0.4159** | **0.4650** | 0.0778 | Full test (5427) |
| RoBERTa-base | Fine-tune | 0.4111 | 0.4618 | 0.0795 | Full test (5427) |
| Qwen2.5 3B Instruct | Zero-shot (offline) | 0.2219 | 0.2717 | **0.0713** | 2K subset |
| Llama 3.2 3B Instruct | Zero-shot (offline) | 0.2126 | 0.2306 | 0.1068 | 2K subset |
| *Paper baseline* | *BERT Fine-tune* | *0.46* | *—* | *—* | *Demszky 2020* |
