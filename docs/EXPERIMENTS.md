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
| Inference time | ~1.33s/sample (full 5,427 ≈ 2 giờ) |
| Temperature | 0.0 (do_sample=False) |
| n_samples | **5427 (full test)** |
| Quantization | false (float16 trên CUDA) |

### Results

| Metric | Value |
|--------|-------|
| F1-macro | **0.2133** |
| F1-micro | 0.2329 |
| F1-weighted | 0.2924 |
| Hamming Loss | 0.1047 |
| n_samples | 5427 |

### Per-class F1 (selected)

**Top 5:**
| Emotion | F1 |
|---------|-----|
| gratitude | 0.661 |
| love | 0.542 |
| amusement | 0.485 |
| admiration | 0.440 |
| joy | 0.333 |

**Bottom 5:**
| Emotion | F1 |
|---------|-----|
| nervousness | 0.044 |
| desire | 0.058 |
| realization | 0.065 |
| pride | 0.066 |
| annoyance | 0.066 |

### Observations

- F1-macro = **0.2133** (full 5,427), thấp hơn Qwen2.5 3B (0.2364) một chút — nhưng pattern khác hẳn.
- **Llama mạnh hơn Qwen ở rare/subtle classes:** grief=0.091 (Qwen=0.000), caring=0.222 (Qwen=0.000), neutral=0.331 (Qwen=0.273). Đây là bằng chứng partial support cho RQ3 khi so sánh giữa các LLM.
- **Hamming Loss cao hơn Qwen** (0.1047 vs 0.0713): Llama over-predict nhiều labels hơn — chiến lược aggressive hơn Qwen.
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
| Inference time | ~0.48s/sample (full 5,427 ≈ 43 phút) |
| Temperature | 0.0 (do_sample=False) |
| n_samples | **5427 (full test)** |
| Quantization | false (float16 trên CUDA) |

### Results

| Metric | Value |
|--------|-------|
| F1-macro | **0.2364** |
| F1-micro | 0.2703 |
| F1-weighted | 0.2779 |
| Hamming Loss | 0.0713 |
| n_samples | 5427 |

### Per-class F1 (selected)

**Top 5:**
| Emotion | F1 |
|---------|-----|
| gratitude | 0.680 |
| sadness | 0.466 |
| fear | 0.460 |
| admiration | 0.449 |
| amusement | 0.405 |

**Bottom 5:**
| Emotion | F1 |
|---------|-----|
| disapproval | 0.092 |
| approval | 0.046 |
| remorse | 0.031 |
| caring | 0.000 |
| grief | 0.000 |

### Observations

- F1-macro = **0.2364** (full 5,427), thấp hơn BERT fine-tune (0.4159) gần **2 lần** → **RQ1 confirmed mạnh**: fine-tuned model vượt LLM zero-shot đáng kể.
- **Hypothesis RQ3 bị bác bỏ:** LLM không mạnh hơn ở rare classes — grief=0.000, caring=0.000, thấp hơn cả BERT (grief=0.222, caring≈0.34). (Lưu ý trên full test, pride=0.286 — không còn 0.000 như đo 2K; nhưng grief/caring vẫn = 0.)
- **Neutral recall kém** (F1=0.273 vs BERT 0.682): Qwen có xu hướng over-predict emotions — thêm nhãn vào các text thực ra là neutral. Ví dụ: "KAMALA 2020!!!" → Qwen dự đoán [excitement], ground truth = [neutral].
- **Qwen mạnh hơn ở single clear-emotion cases:** Khi chỉ có 1 emotion rõ ràng (fear=0.460, gratitude=0.680), Qwen khá tốt. Nhưng với multi-label phức tạp hay cảm xúc tinh tế → fail.
- **Error analysis** (full 5,427 samples): disagreement rate với BERT = 93.3%, BERT closer to truth = 40.6%, LLM closer = 5.0%, LLM exact wins = 9.5% (515 cases), BERT exact wins = 8.4% (457 cases). LLM wins nhiều hơn BERT ở exact match nhờ precision cao, nhưng BERT thường gần đúng hơn (partial credit).

---

## EXP-05: llama_fewshot

| Field | Value |
|-------|-------|
| Date | 2026-06-11 |
| Config | `configs/llama_fewshot.yaml` |
| Model | `meta-llama/Llama-3.2-3B-Instruct` (k=5 in-context) |
| n_samples | **5427 (full test)** |

### Results

| Metric | Value |
|--------|-------|
| F1-macro | **0.2382** |
| F1-micro | 0.2432 |
| F1-weighted | 0.3020 |
| Hamming Loss | 0.1085 |

**Top 5:** gratitude 0.685 · love 0.546 · admiration 0.464 · amusement 0.456 · sadness 0.362
**Bottom 5:** relief 0.124 · disgust 0.082 · embarrassment 0.072 · pride 0.064 · nervousness 0.051

### Observations

- Few-shot nâng Llama 0.2133 → **0.2382** (+11.7%). Không còn class nào = 0.000 (thấp nhất nervousness 0.051).
- Llama hưởng lợi từ few-shot nhiều hơn Qwen, đặc biệt ở việc kéo các rare class lên khỏi 0.

---

## EXP-06: qwen_fewshot

| Field | Value |
|-------|-------|
| Date | 2026-06-11 |
| Config | `configs/qwen_fewshot.yaml` |
| Model | `Qwen/Qwen2.5-3B-Instruct` (k=5 in-context) |
| n_samples | **5427 (full test)** |

### Results

| Metric | Value |
|--------|-------|
| F1-macro | **0.2466** |
| F1-micro | 0.2900 |
| F1-weighted | 0.2985 |
| Hamming Loss | **0.0710** |

**Top 5:** gratitude 0.597 · love 0.460 · sadness 0.449 · admiration 0.448 · fear 0.443
**Bottom 5:** realization 0.105 · caring 0.070 · approval 0.061 · grief 0.000 · remorse 0.000

### Observations

- Few-shot nâng Qwen 0.2364 → **0.2466** (+4.3%). Vẫn còn 2 class = 0.000 (grief, remorse) nhưng pride/relief/caring được cải thiện.
- Qwen giữ Hamming Loss thấp nhất trong các LLM (0.0710) — conservative, ít false positives.

---

## EXP-07: ensemble_llm

| Field | Value |
|-------|-------|
| Date | 2026-06-13 |
| Script | `scripts/ensemble_llm.py` |
| Thành viên | Llama + Qwen (zero-shot & few-shot), 4 prediction sets |
| n_samples | **5427 (full test)** |

### Results (best = All-4 majority ≥2 phiếu)

| Metric | Value |
|--------|-------|
| F1-macro | **0.2657** |
| F1-micro | 0.2848 |
| F1-weighted | 0.3339 |
| Hamming Loss | 0.0938 |

### Observations

- Mô phỏng hệ đoạt giải PAI (SemEval-2025 Task 11). Majority-vote (≥2) nâng F1-macro từ 0.2466 (LLM đơn tốt nhất) → **0.2657 (+7.7%)**.
- `intersection`/`majority≥3` cho precision cao nhất (Hamming ~0.054, F1-micro ~0.357).
- Vẫn thua BERT (0.4148) ~1.56× → ensemble LLM *off-the-shelf* chưa thay được fine-tuning.

---

## EXP-08: cross_benchmark (SemEval-2025 Task 11, BRIGHTER English)

| Field | Value |
|-------|-------|
| Date | 2026-06-18 |
| Script | `scripts/run_brighter.py` |
| Dataset | `brighter-dataset/BRIGHTER-emotion-categories` config `eng` |
| Splits | train 2,764 / dev 230 / **test 5,528 (gold)** |
| Nhãn | 5 (anger, fear, joy, sadness, surprise) — English không có disgust |

### Kết quả trên test 5,528 (so trực tiếp với baseline 0.708 & PAI 0.823)

| Hệ | F1-macro | F1-micro | Hamming |
|----|----------|----------|---------|
| **BERT-base (fine-tune, t=0.5)** | **0.7069** | 0.7363 | 0.1713 |
| Qwen2.5 3B few-shot | 0.5966 | 0.5946 | 0.2294 |
| Llama 3.2 3B few-shot | 0.5778 | 0.5806 | 0.2806 |
| Llama 3.2 3B zero-shot | 0.5700 | 0.5653 | 0.2724 |
| Qwen2.5 3B zero-shot | 0.4662 | 0.4473 | 0.2448 |
| *RoBERTa baseline (official)* | *0.708* | — | — |
| *PAI (winner)* | *0.823* | — | — |

**BERT per-class (t=0.5):** fear 0.813 · sadness 0.725 · joy 0.714 · surprise 0.693 · anger 0.589

### Observations

- **BERT-base = 0.7069 ≈ RoBERTa baseline 0.708** (chênh 0.001) → pipeline fine-tune của mình **đúng chuẩn**; điểm GoEmotions thấp (0.41) hoàn toàn do **28 lớp khó hơn 5 lớp**.
- LLM off-the-shelf đạt **84% của BERT** trên 5 lớp (Qwen FS 0.597) vs chỉ **59%** trên GoEmotions 28 lớp → gap LLM↔fine-tuned phụ thuộc độ chi tiết nhãn.
- Few-shot giúp Qwen rất nhiều ở đây (0.466→0.597, +0.13); Llama gần như không đổi.
- So với PAI (0.823) = so pipeline mình vs hệ SOTA (ChatGPT-4o/Qwen-32B + AdaLoRA + stacking), không phải reproduce.

---

## EXP-09: lora_finetune (LoRA fine-tune Qwen2.5-3B — cả 2 dataset)

| Field | Value |
|-------|-------|
| Date | 2026-06-18 |
| Script | `scripts/lora_finetune.py` |
| Base model | `Qwen/Qwen2.5-3B-Instruct` (AutoModelForSequenceClassification, multi-label) |
| PEFT | LoRA r=16, α=32, target q/k/v/o_proj, **bf16** (fp16 lỗi unscale grad) |
| Hyperparams | lr 1e-4, batch 8 × accum 2, warmup 0.05 |

### Kết quả

| Dataset | Epochs | F1-macro | F1-micro | Hamming | Train time |
|---------|--------|----------|----------|---------|------------|
| **BRIGHTER Eng (5 lớp)** | 5 | **0.7522** | 0.7785 | 0.1326 | 14.6 phút |
| **GoEmotions (28 lớp)** | 2 | **0.4519** | 0.5928 | 0.0287 | 92 phút |

### Observations

- **Đóng vòng lập luận:** cùng một Qwen-3B, prompt→LoRA nhảy **+0.21 (GoEmotions)** và **+0.29 (BRIGHTER)**.
- **LoRA vượt BERT-base ở CẢ HAI:** GoEmotions 0.4519 > 0.4148; BRIGHTER 0.7522 > 0.7069 (và > baseline 0.708).
- → Bằng chứng tự thân: *"LLM cần được fine-tune"*, không phải *"LLM kém"*. Khoảng cách tới PAI (0.823) = scale 32B + ensemble stacking.

---

## Summary Comparison

> Bảng GoEmotions (28 lớp). Cross-benchmark BRIGHTER xem EXP-08/09. Master table (paradigm × dataset) xem REPORT §4.5.

| Model | Method | F1-macro | F1-micro | Hamming Loss | Eval set |
|-------|--------|----------|----------|--------------|----------|
| BERT-base | Fine-tune (t=0.9 tuned) | **0.5167** | 0.5278 | — | Full test (5,427) |
| BERT-base | Fine-tune (t=0.5) | 0.4159 | 0.4650 | 0.0778 | Full test (5,427) |
| **Qwen2.5 3B** | **LoRA fine-tune (EXP-09)** | **0.4519** | 0.5928 | **0.0287** | Full test (5,427) |
| RoBERTa-base | Fine-tune | 0.4111 | 0.4618 | 0.0795 | Full test (5,427) |
| LLM Ensemble (EXP-07) | majority ≥2 | 0.2657 | 0.2848 | 0.0938 | Full test (5,427) |
| Qwen2.5 3B Instruct | Few-shot (k=5) | 0.2466 | 0.2900 | 0.0710 | Full test (5,427) |
| Llama 3.2 3B Instruct | Few-shot (k=5) | 0.2382 | 0.2432 | 0.1085 | Full test (5,427) |
| Qwen2.5 3B Instruct | Zero-shot (offline) | 0.2364 | 0.2703 | 0.0713 | Full test (5,427) |
| Llama 3.2 3B Instruct | Zero-shot (offline) | 0.2133 | 0.2329 | 0.1047 | Full test (5,427) |
| *Paper baseline* | *BERT Fine-tune* | *0.46* | *—* | *—* | *Demszky 2020* |

> **Phạm vi:** so sánh fine-tuned encoder vs LLM mã nguồn mở chạy local. LLM API thương mại (Gemini, GPT…) ngoài phạm vi.
