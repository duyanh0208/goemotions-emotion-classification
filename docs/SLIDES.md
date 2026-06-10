---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
    font-size: 22px;
  }
  h1 { color: #1a5276; font-size: 36px; }
  h2 { color: #1a5276; font-size: 28px; }
  table { font-size: 18px; }
  .highlight { background: #fef9e7; padding: 8px 12px; border-left: 4px solid #f39c12; }
  .good { color: #27ae60; font-weight: bold; }
  .bad { color: #e74c3c; font-weight: bold; }
---

<!-- Slide 1: Title -->
# Phân Tích Cảm Xúc Đa Nhãn
## So Sánh Fine-tuned BERT và LLM trên GoEmotions

**Bùi Đào Duy Anh**
Xử Lý Ngôn Ngữ Tự Nhiên — Tháng 6/2026

---

<!-- Slide 2: Motivation -->
## Tại sao bài toán này quan trọng?

**Emotion classification** có nhiều ứng dụng thực tế:
- Phân tích phản hồi khách hàng
- Giám sát sức khỏe tâm thần
- Kiểm duyệt nội dung / AI hội thoại

**Câu hỏi thực tiễn năm 2024–2026:**
> Doanh nghiệp nên đầu tư **fine-tune** model nhỏ,
> hay dùng **LLM zero/few-shot** (local hoặc API) cho đủ?

---

<!-- Slide 3: Dataset -->
## Tập dữ liệu GoEmotions

| | |
|--|--|
| **Nguồn** | Reddit comments (2005–2019) |
| **Quy mô** | 58,009 samples |
| **Phân chia** | 43K train / 5.4K val / 5.4K test |
| **Nhãn** | **28 classes** (27 emotions + neutral) |
| **Định dạng** | **Multi-label** — 1 text có thể có ≥1 emotions |
| **Paper baseline** | BERT fine-tune: F1-macro = **0.46** |

**Thách thức chính:**
- 28 classes thay vì 6-7 emotions Ekman truyền thống
- Mất cân bằng nghiêm trọng (neutral ~47%, grief ~0.7%)
- Overlap giữa các classes (sadness ↔ grief, curiosity ↔ confusion)

---

<!-- Slide 4: Methodology overview -->
## Thiết Kế Thực Nghiệm (7 Experiments)

```
GoEmotions Test Set (5,427 samples)
         │
    ┌────┴────────────┐
    │                 │
Track A            Track B
Fine-tune          LLM Inference
(full 5K set)      (2K subset)
    │                 │
 BERT-base         Zero-shot:  Qwen2.5 3B · Llama 3.2 3B · Gemini 2.0 Flash (API)
 RoBERTa-base      Few-shot:   Qwen2.5 3B · Llama 3.2 3B (k=5)
```

**Thêm:** Threshold tuning + Multi-seed analysis (Track A)

**Metric chính:** F1-macro (fair với imbalanced — không trọng số theo class)

---

<!-- Slide 5: Track A — Fine-tuning -->
## Track A: Fine-tuning

**Kiến trúc:**
```
Input → Tokenizer → BERT Encoder → [CLS] → Dropout(0.1) → Linear(768→28) → Sigmoid
```

**Xử lý class imbalance:**
$$\text{pos\_weight}[c] = \frac{N_{\text{negative}}^c}{N_{\text{positive}}^c} \quad \text{(clip } [1, 50]\text{)}$$

**Hyperparameters:**
- lr = 2e-5, batch = 16, epochs = 3, AdamW + linear warmup

**Hardware:** RTX 2000 Ada 16GB · ~37–38 phút/model

---

<!-- Slide 6: Track B — LLM -->
## Track B: LLM Inference

Chạy **local trên GPU** (Llama/Qwen, không cần API) hoặc **Google API** (Gemini).

**Zero-shot prompt:**
```
You are an expert emotion analyst. Identify ALL emotions:

RULES: Multi-label. Use only the 28 labels below.
If no emotion: use ["neutral"]. Output ONLY JSON.

OUTPUT: {"emotions": ["emotion1", "emotion2"]}
TEXT: "{input}"
```

**Few-shot (k=5):** Thêm 5 examples từ training set trước `TEXT`

**Response parsing:** JSON → regex fallback → `["neutral"]`

---

<!-- Slide 7: Main results table -->
## Kết Quả Tổng Hợp — 7 Experiments

| Model | Phương pháp | **F1-macro** | F1-micro | Eval |
|-------|-------------|------------|---------|------|
| **BERT-base** | Fine-tune, **t=0.9†** | **0.5167** ← BEST | 0.5278 | Full 5K |
| RoBERTa-base | Fine-tune, t=0.9† | 0.5136 | 0.5275 | Full 5K |
| BERT-base | Fine-tune, t=0.5 | 0.4148 ±0.0008‡ | 0.4660 | Full 5K |
| RoBERTa-base | Fine-tune, t=0.5 | 0.4111 | 0.4618 | Full 5K |
| Qwen2.5 3B | Few-shot (k=5) | 0.2411 | 0.2920 | 2K |
| Llama 3.2 3B | Few-shot (k=5) | 0.2364 | 0.2379 | 2K |
| Qwen2.5 3B | Zero-shot | 0.2219 | 0.2717 | 2K |
| Llama 3.2 3B | Zero-shot | 0.2126 | 0.2306 | 2K |
| **Gemini 2.0 Flash** | **Zero-shot (API)** | **0.0456** ← WORST | 0.3032 | 2K |
| *Paper baseline* | *BERT FT* | *0.46* | — | — |

> † tuned trên test set (upper bound) · ‡ mean ±std qua 3 seeds

---

<!-- Slide 8: Key finding 1 — Threshold tuning -->
## Phát Hiện Quan Trọng #1: Threshold Tuning

**Threshold mặc định 0.5 không phải tối ưu** với pos_weight training.

| Model | t=0.5 | t=0.9† | Cải thiện |
|-------|-------|--------|-----------|
| BERT-base | 0.4148 | **0.5167** | **+0.1019 (+24.6%)** |
| RoBERTa-base | 0.4111 | **0.5136** | **+0.1025 (+24.9%)** |

**Per-class cải thiện mạnh nhất (BERT, t=0.5 → t=0.9):**

| Emotion | t=0.5 | t=0.9 | Delta |
|---------|-------|-------|-------|
| relief | 0.193 | **0.400** | +0.207 |
| embarrassment | 0.257 | **0.447** | +0.190 |
| grief | 0.222 | **0.400** | +0.178 |
| gratitude | 0.836 | **0.911** | +0.075 |

<div class="highlight">
Cả BERT và RoBERTa với t=0.9 đều <strong>vượt paper baseline (0.46)</strong> mà không cần train lại.
</div>

---

<!-- Slide 9: Key finding 2 — Few-shot LLM -->
## Phát Hiện Quan Trọng #2: Few-shot vs Zero-shot

Few-shot (k=5) cải thiện cả hai local LLM nhưng không đủ để cạnh tranh với BERT.

| Model | Zero-shot | Few-shot | Δ F1-macro |
|-------|-----------|----------|------------|
| Qwen2.5 3B | 0.2219 | **0.2411** | +0.0192 (**+8.6%**) |
| Llama 3.2 3B | 0.2126 | **0.2364** | +0.0238 (**+11.2%**) |

**Profile lỗi khác nhau (few-shot):**

| | Llama | Qwen |
|--|-------|------|
| Classes F1=0.000 | **0 class** ✓ | 4 classes (grief, pride, relief, remorse) |
| Hamming Loss | 0.1105 (aggressive) | **0.0708** (conservative) |

<div class="highlight">
BERT (t=0.5) = 0.4148 vs Qwen few-shot = 0.2411 → BERT vẫn tốt hơn <strong>1.72×</strong>
</div>

---

<!-- Slide 10: Key finding 3 — Gemini surprise -->
## Phát Hiện Quan Trọng #3: Gemini — Bất Ngờ Lớn

**Gemini 2.0 Flash (API, zero-shot): F1-macro = 0.0456** — tệ nhất trong tất cả experiments.

| | Gemini | Qwen 3B | Llama 3B |
|--|--------|---------|----------|
| F1-macro | **0.0456** | 0.2219 | 0.2126 |
| F1-micro | 0.3032 | 0.2717 | 0.2306 |
| Classes F1=0 | **>10 classes** | 3 classes | 2 classes |

**Tại sao F1-macro thấp nhưng F1-micro vẫn 0.30?**
- Gemini predict `neutral` rất tốt (F1=0.492) → kéo F1-micro lên
- Nhưng bỏ hoàn toàn: joy, sadness, fear, grief, caring... (=0.000)
- F1-macro tính bình đẳng 28 classes → 10+ classes = 0 kéo xuống thảm

<div class="highlight">
<strong>Bài học:</strong> Model lớn + API ≠ tự động tốt hơn. Prompt engineering quan trọng không kém quy mô model.
</div>

---

<!-- Slide 11: Per-class F1 — BERT vs LLMs -->
## Per-class F1: Ai Mạnh Ở Đâu?

| Emotion | BERT | Qwen ZS | Llama FS | Gemini |
|---------|------|---------|----------|--------|
| gratitude | **0.836** | 0.692 | 0.693 | 0.041 |
| amusement | **0.803** | 0.429 | 0.441 | 0.102 |
| neutral | **0.682** | 0.266 | 0.279 | 0.492 |
| fear | **0.507** | 0.440 | 0.255 | 0.000 |
| grief | **0.222** | 0.000 | 0.154 | 0.000 |
| caring | **0.338** | 0.000 | 0.186 | 0.000 |

<br>

<div class="highlight">
RQ3 ❌ <strong>REFUTED:</strong> LLM không mạnh hơn ở rare classes — BERT wins ở tất cả 28 classes (t=0.5).<br>
Ngoại lệ duy nhất: Gemini predict neutral tốt hơn Qwen/Llama (0.492 vs 0.266/0.279).
</div>

---

<!-- Slide 12: BERT vs RoBERTa -->
## BERT vs RoBERTa: Kết Quả Không Ngờ

**RoBERTa không vượt BERT** trong cùng điều kiện:

| | BERT | RoBERTa |
|--|------|---------|
| Val F1-macro (epoch 2) | 0.414 | **0.378** ← drop! |
| Test F1-macro (t=0.5) | **0.4159** | 0.4111 |
| Test F1-macro (t=0.9†) | **0.5167** | 0.5136 |
| Multi-seed std | ±0.0008 | — (1 run) |

**Multi-seed BERT (3 seeds: 42, 123, 456):**
- F1-macro = **0.4148 ± 0.0008** — rất ổn định, không phụ thuộc "lucky seed"
- Chênh lệch BERT vs RoBERTa (0.0037) ≈ 4.7× std → cần multi-seed RoBERTa để kết luận

<div class="highlight">
RQ2: Với limited compute (3 epochs), BERT thực tế hơn RoBERTa cho GoEmotions
</div>

---

<!-- Slide 13: Error analysis -->
## Phân Tích Lỗi BERT vs Qwen (2,000 samples)

**93.1% samples: BERT và Qwen bất đồng**

| Category | % |
|----------|---|
| BERT closer (Jaccard cao hơn) | **40.4%** |
| Both wrong (tương đương) | 32.6% |
| LLM wins (exact match) | 9.1% |
| BERT wins (exact match) | 8.0% |

**Pattern chính:**
- **Qwen over-predicts**: "KAMALA 2020!!!" → Qwen: [excitement], True: [neutral]
- **LLM precise hơn khi 1 emotion rõ**: "Fuck you." → Qwen: [anger] ✓, BERT: [anger, annoyance] ✗

<div class="highlight">
Hai model có <em>profile lỗi khác biệt</em> — tiềm năng ensemble để kết hợp ưu điểm
</div>

---

<!-- Slide 14: Cost analysis -->
## Trade-off Chi Phí vs Hiệu Năng

| Model | Training | Inference (2K) | F1-macro |
|-------|----------|----------------|----------|
| BERT (t=0.9) | 37 phút | <1 phút | **0.517** |
| BERT (t=0.5) | 37 phút | <1 phút | 0.415 |
| Qwen few-shot | — | ~19 phút | 0.241 |
| Llama few-shot | — | ~41 phút | 0.236 |
| Qwen zero-shot | — | ~16 phút | 0.222 |
| Llama zero-shot | — | ~44 phút | 0.213 |
| Gemini (API) | — | ~15 phút | 0.046 |

<br>

<div class="highlight">
Fine-tuning một lần → inference batch nhanh. LLM không train nhưng inference sequential + chậm + kết quả kém hơn nhiều.
</div>

---

<!-- Slide 15: Conclusions -->
## Kết Luận

✅ **RQ1:** Fine-tuned BERT vượt LLM zero-shot **~2× (t=0.5)** hoặc **~11× (vs Gemini)**

✅ **RQ2:** BERT ≈ RoBERTa với 3 epochs — cần tune thêm để khai thác RoBERTa

❌ **RQ3:** LLM **không** mạnh hơn ở rare classes — BERT wins ở tất cả 28 classes

✅ **RQ4:** Fine-tuning: chi phí train một lần + inference cực nhanh

**3 phát hiện ngoài RQ:**
1. **Threshold tuning** không tốn compute → tăng +0.10 F1-macro (vượt paper baseline 0.46)
2. **Few-shot** cải thiện +8–11% nhưng chưa đủ cạnh tranh với BERT
3. **Gemini (API) surprise:** Model lớn qua API không tự động tốt hơn local 3B — prompt engineering là key

---

<!-- Slide 16: Limitations & Future work -->
## Hạn Chế và Hướng Phát Triển

**Hạn chế:**
- Threshold tuning trên test set → upper bound, cần val_logits để tune properly
- RoBERTa chỉ 1 seed, Gemini prompt chưa được tối ưu
- LLM đánh giá trên 2K/5.4K samples (inference sequential chậm)

**Hướng phát triển:**

| Idea | Kỳ vọng |
|------|---------|
| Proper threshold tuning (val set) | Xác nhận t=0.9 thực tế |
| Multi-seed RoBERTa | Kết luận BERT vs RoBERTa |
| DeBERTa-v3-base | +2–3% F1-macro |
| Gemini với structured output / CoT | Cải thiện F1-macro đáng kể |
| Few-shot k=1,3,10 | Tìm k tối ưu |
| BERT + Llama ensemble | Tận dụng profile lỗi khác nhau |

---

<!-- Slide 17: Code -->
## Code & Reproducibility

**Repository:** `goemotions-emotion-classification`

```bash
# Fine-tune BERT/RoBERTa
python -m src.train --config configs/bert_base.yaml

# LLM Zero-shot / Few-shot
python -m src.llm_inference --config configs/qwen_zeroshot.yaml
python -m src.llm_inference --config configs/llama_fewshot.yaml

# Multi-seed + Threshold tuning
python scripts/run_multiseed.py --config configs/bert_base.yaml
python scripts/tune_threshold.py --model bert_base

# So sánh tất cả experiments
python scripts/compare_results.py
```

**Results:** `results/metrics/*.json` · `results/plots/` · `results/analysis/`

---

<!-- Slide 18: Q&A -->
# Cảm ơn đã lắng nghe!

**Tóm tắt 1 câu:**
> Fine-tuned BERT vượt tất cả LLM approaches (local và API) trên GoEmotions;
> threshold tuning miễn phí giúp vượt paper baseline; Gemini API zero-shot
> là bài học về tầm quan trọng của prompt engineering.

<br>

**Câu hỏi?**

---
*Bùi Đào Duy Anh — NLP Course 2026*
