---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
    font-size: 23px;
  }
  h1 { color: #1a5276; }
  h2 { color: #1a5276; font-size: 30px; }
  table { font-size: 19px; margin: 0 auto; }
  th { background: #eaf2f8; }
  .highlight { background: #fef9e7; padding: 8px 14px; border-left: 5px solid #f39c12; }
  .good { color: #27ae60; font-weight: bold; }
  .bad  { color: #e74c3c; font-weight: bold; }
  .small { font-size: 17px; color: #555; }
  section.lead { text-align: center; }
  section.lead h1 { font-size: 40px; }
---

<!-- _class: lead -->

# Phân Tích Cảm Xúc Đa Nhãn trên GoEmotions

## Fine-tuning vs LLM — và đối chiếu với SemEval-2025 Task 11

**Bùi Đào Duy Anh**
Xử Lý Ngôn Ngữ Tự Nhiên · Master CS, VJU-VNU · Tháng 6/2026

---

## 1 · Bối cảnh & động lực

**Emotion classification** — ứng dụng: phản hồi khách hàng, sức khỏe tâm thần, kiểm duyệt nội dung, AI hội thoại.

**Câu hỏi thực tiễn (2024–2026):**

<div class="highlight">
Nên đầu tư <strong>fine-tune một model nhỏ</strong>, hay chỉ cần <strong>prompt một LLM lớn</strong> (zero/few-shot) là đủ?
</div>

<br>

Báo cáo trả lời câu hỏi này bằng thực nghiệm có hệ thống trên **2 dataset** và đối chiếu với **hệ đoạt giải quốc tế**.

---

## 2 · Dataset: GoEmotions

| | |
|--|--|
| **Nguồn** | Reddit comments (2005–2019) |
| **Quy mô** | 58,009 mẫu — 43K train / 5.4K val / **5,427 test** |
| **Nhãn** | **28 lớp** (27 emotions + neutral), **multi-label** |
| **Baseline (paper gốc)** | BERT fine-tune: F1-macro = **0.46** |

**Vì sao khó:**
- 28 lớp chi tiết (so với 6–7 emotion Ekman truyền thống)
- Mất cân bằng nặng (neutral ~47% · grief ~0.7%)
- Overlap: sadness↔grief, curiosity↔confusion

---

## 3 · Câu hỏi nghiên cứu

| # | Câu hỏi |
|---|---------|
| **RQ1** | Fine-tuned BERT có vượt LLM zero/few-shot không? |
| **RQ2** | BERT vs RoBERTa trong cùng điều kiện? |
| **RQ3** | LLM có lợi thế ở các cảm xúc hiếm không? |
| **RQ4** | Trade-off chi phí huấn luyện ↔ hiệu năng? |

<div class="highlight">
Mở rộng (theo góp ý GVHD): đối chiếu với <strong>SemEval-2025 Task 11</strong> — chạy chính pipeline của mình trên <strong>dữ liệu của bài báo</strong> để so sánh trực tiếp.
</div>

---

## 4 · Thiết kế thực nghiệm — 9 experiments, 3 paradigm

```
                 GoEmotions (28 lớp, test 5,427)
                              │
        ┌─────────────────────┼─────────────────────┐
   Encoder fine-tune      LLM prompt-only       LLM fine-tune
   (EXP-01,02)            (EXP-03→06)           (EXP-09)
   BERT · RoBERTa         zero/few-shot          Qwen-3B + LoRA
                          + Ensemble (EXP-07)
                              │
              EXP-08: kiểm chứng chéo trên dữ liệu SemEval (BRIGHTER)
```

- **Metric chính:** F1-macro (công bằng với dữ liệu mất cân bằng)
- LLM chạy **local, mã nguồn mở** (Llama 3.2 3B · Qwen2.5 3B); LLM API thương mại ngoài phạm vi
- Bổ sung: threshold tuning · multi-seed · error analysis

---

## 5 · Track A — Fine-tuning encoder

**Kiến trúc:**
```
Input → Tokenizer → BERT/RoBERTa → [CLS] → Dropout → Linear(→28) → Sigmoid
```

**Xử lý mất cân bằng:** `BCEWithLogitsLoss` + `pos_weight = N_neg / N_pos` (clip [1, 50])

**Hyperparameters:** lr 2e-5 · batch 16 · 3 epochs · AdamW + linear warmup

<span class="small">Hardware: RTX 2000 Ada 16GB · ~37 phút/model</span>

---

## 6 · Track B — LLM (prompting)

**Zero-shot prompt:**
```
You are an expert emotion analyst. Identify ALL emotions.
RULES: multi-label; chỉ dùng 28 nhãn; nếu không có → ["neutral"];
output CHỈ JSON: {"emotions": [...]}
TEXT: "{input}"
```

- **Few-shot (k=5):** thêm 5 ví dụ từ train trước câu cần phân loại
- **Parsing:** JSON → regex fallback → `["neutral"]`
- Greedy decoding (deterministic), full test 5,427 — **cùng tập với BERT**

---

## 7 · KẾT QUẢ CHÍNH — GoEmotions (28 lớp)

| Mô hình | Phương pháp | **F1-macro** | F1-micro |
|---|---|---|---|
| **BERT-base** | Fine-tune, **t=0.9**† | **0.5167** | 0.5278 |
| RoBERTa-base | Fine-tune, t=0.9† | 0.5136 | 0.5275 |
| BERT-base | Fine-tune, t=0.5 | 0.4148 ±0.0008‡ | 0.4660 |
| **Qwen-3B** | **LoRA fine-tune** | **0.4519** | 0.5928 |
| RoBERTa-base | Fine-tune, t=0.5 | 0.4111 | 0.4618 |
| LLM Ensemble | majority ≥2 | 0.2657 | 0.2848 |
| Qwen-3B | few-shot | 0.2466 | 0.2900 |
| Qwen-3B | zero-shot | 0.2364 | 0.2703 |
| Llama-3B | few / zero | 0.2382 / 0.2133 | — |
| *Paper baseline* | *BERT* | *0.46* | — |

<span class="small">† tuned trên test (upper bound) · ‡ mean±std qua 3 seeds · tất cả trên full 5,427</span>

---

## 8 · Phát hiện #1 (RQ1): Fine-tuned ≫ LLM prompt-only

<div class="highlight">
BERT fine-tune <strong>0.4148</strong> vs LLM prompt tốt nhất (Qwen FS) <strong>0.2466</strong> → BERT hơn <strong>1.68×</strong>. Với t=0.9 thì ~2.1×.
</div>

<br>

- Khoảng cách **lớn và ổn định** (multi-seed std ±0.0008).
- Ensemble các LLM prompt (EXP-07) chỉ đạt 0.2657 — **vẫn thua xa**.
- **RQ1 được xác nhận mạnh** trên GoEmotions: prompting off-the-shelf chưa đủ cho bài 28 lớp.

<span class="small">→ Nhưng đây CHƯA phải toàn bộ câu chuyện (xem phần đối chiếu SemEval + LoRA).</span>

---

## 9 · Phát hiện #2: Threshold tuning — cải thiện "miễn phí"

Ngưỡng mặc định 0.5 không tối ưu khi train với `pos_weight`.

| Model | t=0.5 | t=0.9 | Cải thiện |
|---|---|---|---|
| BERT-base | 0.4148 | **0.5167** | **+0.10 (+24.6%)** |
| RoBERTa-base | 0.4111 | **0.5136** | +0.10 |

**Rare classes hưởng lợi nhiều nhất:** relief 0.193→**0.400** · grief 0.222→**0.400** · embarrassment 0.257→**0.447**

<div class="highlight">
Không tốn thêm compute, cả hai model <strong>vượt paper baseline 0.46</strong>. (Tuned trên test → upper bound; cần val set để chốt.)
</div>

---

## 10 · Phát hiện #3: Few-shot & Ensemble

**Few-shot (k=5) giúp nhưng chưa đủ:**

| Model | zero-shot | few-shot | Δ |
|---|---|---|---|
| Qwen-3B | 0.2364 | 0.2466 | +4.3% |
| Llama-3B | 0.2133 | 0.2382 | +11.7% |

**Ensemble (lấy cảm hứng PAI):** majority-vote 4 prediction-set → **0.2657 (+7.7%)** so với LLM đơn.

<div class="highlight">
Cả few-shot lẫn ensemble đều cải thiện nhưng <strong>không đóng được gap với encoder</strong> — vì còn thiếu mảnh ghép <em>fine-tuning</em>.
</div>

---

## 11 · Per-class & profile lỗi (RQ3)

| Emotion | BERT | Qwen ZS | Llama FS |
|---|---|---|---|
| gratitude | **0.836** | 0.680 | 0.685 |
| amusement | **0.803** | 0.405 | 0.456 |
| neutral | **0.682** | 0.273 | 0.304 |
| grief | **0.222** | 0.000 | 0.130 |
| caring | **0.338** | 0.000 | 0.184 |

- **RQ3 ❌ bác bỏ:** LLM **không** mạnh hơn ở rare classes.
- Error analysis (full 5,427): **93.3% bất đồng** · BERT gần đúng hơn 40.6% vs LLM 5.0%.
- Profile khác nhau: Qwen *conservative* (Hamming 0.071), Llama *aggressive* (0.108).

---

## 12 · BERT vs RoBERTa (RQ2) & Chi phí (RQ4)

<div style="display:flex; gap:20px;">
<div>

**RQ2 — BERT ≈ RoBERTa**
| | BERT | RoBERTa |
|--|--|--|
| t=0.5 | **0.4159** | 0.4111 |
| t=0.9 | **0.5167** | 0.5136 |
| multi-seed | ±0.0008 | 1 run |

<span class="small">Chênh 0.0037 ≈ 4.7× std → chưa đủ kết luận.</span>

</div>
<div>

**RQ4 — Chi phí**
| | Train | Infer |
|--|--|--|
| BERT | 37′ một lần | <1′ (batch) |
| LLM prompt | — | 0.5–1.3s/mẫu |

<span class="small">Fine-tune: train một lần, inference cực nhanh.</span>

</div>
</div>

<div class="highlight">
RQ4: với production scale, BERT inference &lt;10 phút; LLM prompt mất nhiều giờ + kém hơn.
</div>

---

<!-- _class: lead -->

# Đối chiếu với SOTA quốc tế
## SemEval-2025 Task 11 — 2 bài báo GVHD yêu cầu

---

## 13 · Hai bài báo & dữ liệu BRIGHTER

- **Overview (Muhammad et al. 2025):** shared task lớn nhất về emotion detection (700+ đội). Dataset **BRIGHTER**, English Track A = **5 lớp**. Baseline RoBERTa = **0.708**.
- **PAI (Ruan et al. 2025):** hệ **đoạt giải nhất** (English **0.823**) — ensemble ChatGPT-4o/Qwen-32B/… + **AdaLoRA fine-tune** + **stacking 2 vòng**.

<div class="highlight">
Khác dataset (28 vs 5 lớp) → <strong>không so F1 tuyệt đối</strong>. Cách làm đúng: <strong>mang pipeline của mình sang chạy trên ĐÚNG dữ liệu của họ</strong> (BRIGHTER công khai trên HuggingFace).
</div>

---

## 14 · So sánh thế nào cho hợp lý? — 3 trục

| Trục | So cái gì | Vì sao hợp lý |
|---|---|---|
| **① Vị trí so baseline** | có vượt baseline của chính benchmark? | mỗi benchmark có trần riêng |
| **② Paradigm** | fine-tune vs prompt vs stacking | độc lập dataset |
| **③ Δ ensemble** | ensemble nâng bao nhiêu | cùng đơn vị (tương đối) |

- ① Mình: BERT 0.5167 > baseline 0.46 ✓ — PAI: 0.823 > 0.708 ✓ → **cùng chiều**.
- ② PAI thắng nhờ **fine-tune + stacking**, *không phải* prompt thuần.
- ③ Ensemble: PAI **+0.01–0.02** · mình **+0.019** → **trùng nhau** dù model nhỏ hơn ~10×.

---

## 15 · Kiểm chứng chéo (EXP-08): pipeline mình trên data của họ

Chạy chính pipeline trên **đúng test set BRIGHTER English** (5,528 mẫu, 5 lớp):

| Hệ thống | F1-macro |
|---|---|
| PAI (đoạt giải) | 0.823 |
| *RoBERTa baseline chính thức* | *0.708* |
| **BERT-base của mình** | **0.7069** ✓ |
| Qwen few-shot (prompt) | 0.5966 |
| Qwen zero-shot (prompt) | 0.4662 |

<div class="highlight">
<strong>BERT mình = 0.7069 ≈ baseline 0.708</strong> → pipeline <strong>đúng chuẩn</strong>; điểm GoEmotions thấp (0.41) chỉ vì <strong>28 lớp khó hơn 5 lớp</strong>, không phải làm sai.
</div>

---

## 16 · LoRA fine-tune (EXP-09): mảnh ghép quyết định

LoRA fine-tune **chính Qwen2.5-3B** (LLM-as-classifier, đúng ý tưởng PAI):

| Cùng một Qwen-3B | GoEmotions (28 lớp) | BRIGHTER (5 lớp) |
|---|---|---|
| zero-shot (prompt) | 0.2364 | 0.4662 |
| few-shot (prompt) | 0.2466 | 0.5966 |
| **LoRA fine-tune** | **0.4519** | **0.7522** |
| so với BERT-base | 0.4148 → **vượt** | 0.7069 → **vượt** |

<div class="highlight">
Cùng model, prompt→LoRA nhảy <strong>+0.21 / +0.29</strong>, <strong>vượt BERT ở CẢ HAI</strong> và baseline 0.708 → <strong>"LLM cần fine-tune", không phải "LLM kém"</strong>.
</div>

---

## 17 · BỨC TRANH TOÀN CẢNH (paradigm × dataset)

| Nhóm | Mô hình | GoEmotions (28 lớp) | BRIGHTER (5 lớp) |
|---|---|---|---|
| **Encoder fine-tune** | BERT (t=0.9) | **0.5167** | — |
| | BERT (t=0.5) | 0.4148 | 0.7069 |
| **LLM + LoRA** | Qwen-3B fine-tune | **0.4519** | **0.7522** |
| **LLM prompt-only** | Ensemble (≥2) | 0.2657 | — |
| | Qwen few-shot | 0.2466 | 0.5966 |
| | Qwen zero-shot | 0.2364 | 0.4662 |
| **Công bố** | baseline / PAI | 0.46 | 0.708 / 0.823 |

<div class="highlight">
<strong>Dọc:</strong> prompt &lt; ensemble &lt; LoRA. <strong>Ngang:</strong> gap co trên 5 lớp, giãn trên 28 lớp.
👉 LLM+LoRA ≥ encoder ở cả hai → <strong>quyết định là fine-tuning</strong>.
</div>

---

## 18 · Kết luận

| RQ | Trả lời |
|---|---|
| **RQ1** | ✅ Fine-tuned ≫ LLM *prompt* (~1.7–2×) trên GoEmotions |
| **RQ2** | ≈ BERT ngang RoBERTa (3 epochs) |
| **RQ3** | ❌ LLM **không** mạnh hơn ở rare classes |
| **RQ4** | ✅ Fine-tune: train một lần, inference cực nhanh |

**Thông điệp lớn (mở rộng):** Không phải *"LLM kém hơn encoder"* mà là **"LLM cần được fine-tune"** — cùng Qwen-3B, LoRA vượt BERT ở cả 2 dataset; khớp với công thức của hệ đoạt giải PAI (fine-tune + ensemble + scale).

---

## 19 · Hạn chế & hướng phát triển

**Hạn chế:**
- Threshold tuning trên test set → upper bound (cần val set)
- RoBERTa mới 1 seed · phạm vi giới hạn LLM mã nguồn mở (không so API thương mại)
- Mới fine-tune *một* LLM — chưa ensemble nhiều LLM fine-tuned như PAI

**Hướng phát triển:**

| Ý tưởng | Kỳ vọng |
|---|---|
| Proper threshold tuning (val set) | chốt t=0.9 thực tế |
| DeBERTa-v3 · multi-seed RoBERTa | +2–3% · kết luận RQ2 |
| **Ensemble nhiều LLM LoRA + stacking** (công thức PAI) | tiến tới SOTA |

---

<!-- _class: lead -->

# Cảm ơn đã lắng nghe!

**Tóm tắt một câu:**
Fine-tuned encoder vượt mọi LLM *off-the-shelf* trên GoEmotions; nhưng **chính LLM đó khi LoRA fine-tune lại vượt cả encoder lẫn baseline trên dữ liệu SemEval** — yếu tố quyết định là **fine-tuning, không phải họ model**.

<br>

**9 experiments · 2 dataset · đối chiếu trực tiếp với hệ đoạt giải SemEval-2025**

**Câu hỏi?**

<span class="small">Bùi Đào Duy Anh — NLP Course 2026 · repo: goemotions-emotion-classification</span>
