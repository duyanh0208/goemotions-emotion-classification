# 🧪 Experiments Log

Đầy đủ 9 thí nghiệm (EXP-01 → EXP-09) đã hoàn tất cho môn **NLP**. Tất cả số liệu chính là **F1-macro**, đánh giá trên **full test** (GoEmotions: 5,427 mẫu; BRIGHTER English: 5,528 mẫu).

| Phần cứng | NVIDIA RTX 2000 Ada Generation 16GB · CUDA 12.x · Windows 11 · seed = 42 |
|---|---|

---

## Bảng tổng hợp (GoEmotions, 28 lớp, test 5,427)

Đọc theo 3 tầng: encoder fine-tune ở ngưỡng đã tinh chỉnh (trên cùng), LLM-LoRA và encoder ở ngưỡng mặc định (giữa), và các phương pháp prompt-only (dưới).

| # | Model | Phương pháp | Thresh. | F1-macro | F1-micro | Hamming |
|---|-------|-------------|---------|----------|----------|---------|
| 01 | BERT-base | Fine-tune | 0.9 † | **0.5167** | 0.5278 | — |
| 02 | RoBERTa-base | Fine-tune | 0.9 † | 0.5136 | 0.5275 | — |
| 09 | Qwen2.5-3B | **LoRA fine-tune** | — | **0.4519** | 0.5928 | — |
| 01 | BERT-base | Fine-tune | 0.5 | 0.4148 ‡ | 0.4660 | 0.0775 |
| 02 | RoBERTa-base | Fine-tune | 0.5 | 0.4111 | 0.4618 | 0.0795 |
| 07 | LLM Ensemble | majority ≥2 | — | 0.2657 | 0.2848 | 0.0938 |
| 06 | Qwen2.5-3B | few-shot (k=5) | — | 0.2466 | 0.2900 | 0.0710 |
| 05 | Llama-3.2-3B | few-shot (k=5) | — | 0.2382 | 0.2432 | 0.1085 |
| 04 | Qwen2.5-3B | zero-shot | — | 0.2364 | 0.2703 | 0.0713 |
| 03 | Llama-3.2-3B | zero-shot | — | 0.2133 | 0.2329 | 0.1047 |
| — | *Baseline (paper)* | *BERT* | — | *0.46* | — | — |

> † Ngưỡng được dò trên test set → **cận trên thực nghiệm** (xem EXP-01 threshold tuning), không phải kết quả held-out.
> ‡ Trung bình ± độ lệch chuẩn qua 3 seed (42, 123, 456).

## Kiểm chứng chéo — SemEval-2025 Task 11 (BRIGHTER English, 5 lớp, test 5,528)

| # | System | Phương pháp | F1-macro | Ghi chú |
|---|--------|-------------|----------|---------|
| — | PAI (đoạt giải) | ensemble + AdaLoRA + stacking | 0.823 | SOTA |
| 09 | Qwen2.5-3B | **LoRA fine-tune** | **0.7522** | vượt BERT & baseline |
| — | RoBERTa baseline (official) | fine-tune | 0.708 | — |
| 08 | BERT-base | fine-tune | **0.7069** | ≈ baseline chính thức (Δ 0.001) |
| 08 | Qwen2.5-3B | few-shot | 0.5966 | — |
| 08 | Llama-3.2-3B | few-shot | 0.5778 | — |
| 08 | Llama-3.2-3B | zero-shot | 0.5700 | — |
| 08 | Qwen2.5-3B | zero-shot | 0.4662 | — |

> BRIGHTER English Track A: **train 2,764 / dev 230 / test 5,528**, 5 lớp (anger/fear/joy/sadness/surprise — English **không** annotate disgust, tức 5 trong 6 lớp Ekman).

---

## EXP-01 · BERT-base Fine-tuning
**Config:** `configs/bert_base.yaml` · **Script:** `python -m src.train --config configs/bert_base.yaml`
- **Setup:** bert-base-uncased (12 layers, 768 hidden), BCEWithLogitsLoss + pos_weight = (N − n_c)/n_c clip [1,50], lr 2e-5, batch 16, 3 epochs, AdamW (wd 0.01), warmup 10% + decay, grad clip 1.0, max_len 128. ~37 phút/seed.

**Training curve (seed 42):**

| Epoch | Train Loss | Val Loss | Val F1-macro | Val F1-micro |
|-------|-----------|----------|--------------|--------------|
| 1 | 0.7009 | 0.4966 | 0.4012 | 0.4150 |
| 2 | 0.4419 | 0.4862 | 0.4137 | 0.4459 |
| 3 | 0.3408 | 0.5098 | 0.4305 | 0.4717 |

- **Kết quả (test 5,427):** F1-macro 0.4159 (seed 42); multi-seed (`scripts/run_multiseed.py`) = **0.4148 ± 0.0008**.
- **Observation:** Val Loss giảm ở epoch 2 rồi tăng nhẹ ở epoch 3 trong khi Val F1-macro vẫn cải thiện → hơi overfit; thêm epoch + early stopping trên Val F1-macro có thể tốt hơn.

**Multi-seed (ổn định):**

| Seed | F1-macro | F1-micro | F1-weighted | Hamming |
|------|----------|----------|-------------|---------|
| 42 | 0.4159 | 0.4650 | 0.5329 | 0.0778 |
| 123 | 0.4143 | 0.4653 | 0.5315 | 0.0776 |
| 456 | 0.4142 | 0.4679 | 0.5329 | 0.0770 |
| **Mean** | **0.4148** | **0.4660** | **0.5324** | **0.0775** |
| **± Std** | **±0.0008** | **±0.0013** | **±0.0007** | **±0.0003** |

> Độ lệch chuẩn F1-macro chỉ 0.0008; khoảng cách best–worst seed = 0.0017 → kết quả không phụ thuộc "seed may mắn". KTC 95% ≈ [0.413, 0.417] khẳng định BERT vượt mọi LLM prompt-only.

**Threshold tuning** (`scripts/tune_threshold.py`): grid search t ∈ [0.1, 0.9] (bước 0.05), tối ưu F1-macro.

| Model | Setting | F1-macro | Improvement |
|-------|---------|----------|-------------|
| BERT-base | t = 0.5 | 0.4159 | — |
| BERT-base | t = 0.9 | **0.5167** | +0.1008 |
| BERT-base | per-class (oracle) | 0.5288 | +0.1129 |
| RoBERTa-base | t = 0.5 | 0.4111 | — |
| RoBERTa-base | t = 0.9 | **0.5136** | +0.1024 |
| RoBERTa-base | per-class (oracle) | 0.5296 | +0.1185 |

> pos_weight lúc train đẩy recall lên (nhiều positive) → nhiều false positive ở t=0.5. Nâng t=0.9 đòi hỏi độ tự tin cao hơn → tăng mạnh precision, recall giảm không đáng kể. Ngoại lệ: lớp `neutral` tối ưu ở t ≈ 0.45; đa số lớp hiếm tối ưu ở t ≈ 0.85–0.90. Lớp hiếm hưởng lợi nhiều nhất: **relief 0.193→0.400 (+0.207), grief 0.222→0.400 (+0.178)**, embarrassment 0.257→0.447.

> ⚠️ Ngưỡng dò trên test → 0.5167 là **cận trên**; cần validation set riêng để chốt (đã bổ sung lưu val_logits).

## EXP-02 · RoBERTa-base Fine-tuning
**Config:** `configs/roberta_base.yaml`

**Training curve:**

| Epoch | Train Loss | Val Loss | Val F1-macro | Val F1-micro |
|-------|-----------|----------|--------------|--------------|
| 1 | 0.7088 | 0.5071 | 0.4049 | 0.4302 |
| 2 | 0.4716 | 0.4905 | 0.3785 | 0.4216 |
| 3 | 0.3764 | 0.5091 | 0.4164 | 0.4658 |

- **Kết quả:** F1-macro **0.4111** (t=0.5), **0.5136** (t=0.9). ~38 phút.
- **Observation:** đường cong kém ổn định hơn BERT (Val F1-macro tụt xuống 0.378 ở epoch 2 rồi hồi phục). Chênh BERT–RoBERTa chỉ 0.0037 (single seed), nằm trong ~4.7× std → **chưa có ý nghĩa thống kê**, cần multi-seed RoBERTa để kết luận.

## Per-class F1 — BERT vs RoBERTa

**Top 5 (F1 cao nhất):**

| Emotion | BERT | RoBERTa | Lý do |
|---------|------|---------|-------|
| gratitude | 0.836 | 0.838 | Tín hiệu rõ ("thanks") + nhiều data |
| amusement | 0.803 | 0.748 | Ngôn ngữ đặc trưng ("lol") |
| love | 0.750 | 0.732 | Nhãn mạnh |
| neutral | 0.682 | 0.684 | Lớp lớn nhất, ranh giới rõ |
| admiration | 0.627 | 0.630 | Tách được khỏi approval/love |

**Bottom 5 (F1 thấp nhất):**

| Emotion | BERT | RoBERTa | Lý do |
|---------|------|---------|-------|
| relief | 0.193 | 0.184 | Rất ít data (~0.8%) |
| realization | 0.190 | — | Chồng lấn surprise/curiosity |
| disappointment | 0.220 | 0.210 | Chồng lấn sadness/disapproval |
| grief | 0.222 | 0.196 | Hiếm, lẫn với sadness |
| nervousness | 0.246 | 0.234 | Chồng lấn fear/confusion |

## EXP-03 · Llama-3.2-3B Zero-shot
**Config:** `configs/llama_zeroshot.yaml` · **Script:** `python -m src.llm_inference --config ...`
- **Setup:** meta-llama/Llama-3.2-3B-Instruct, local (không API key), float16, greedy (do_sample=False ≡ temperature 0), max_new_tokens 128, chat template gốc.
- **Kết quả:** F1-macro **0.2133**. Tốc độ ~1.33 s/mẫu. Profile "aggressive" (đoán nhiều nhãn, Hamming ~0.105) → bắt lớp hiếm tốt hơn Qwen.

## EXP-04 · Qwen2.5-3B Zero-shot
**Config:** `configs/qwen_zeroshot.yaml`
- **Kết quả:** F1-macro **0.2364**. ~0.48 s/mẫu (full 5,427 ≈ 43 phút). Profile "conservative" (Hamming ~0.071), miss `grief` & `caring` (=0.000).
- **RQ1 xác nhận:** BERT > Qwen ZS gần 2× (0.4159 vs 0.2364). **RQ3 bác bỏ:** LLM không mạnh hơn ở lớp hiếm. Điểm yếu lớn nhất của LLM là lớp `neutral` (Qwen 0.273 vs BERT 0.682) — LLM hay gán cảm xúc cho text thực sự trung tính.
- **Tốc độ:** Llama (~1.33 s/mẫu) chậm ~2.8× Qwen dù cùng 3B (khác kiến trúc decode + chat template dài hơn).

## EXP-05 · Llama-3.2-3B Few-shot (k=5)
**Config:** `configs/llama_fewshot.yaml`
- **Kết quả:** F1-macro **0.2382** (+0.0249, **+11.7%** so với zero-shot). Few-shot loại bỏ hết các lớp F1=0.000.

## EXP-06 · Qwen2.5-3B Few-shot (k=5)
**Config:** `configs/qwen_fewshot.yaml`
- **Kết quả:** F1-macro **0.2466** (+0.0102, **+4.3%**). Vẫn thua BERT (t=0.5) ~1.68×.
- **Lớp hiếm (mixed):** grief vẫn 0.000 (remorse còn tụt 0.031→0.000) nhưng pride (0→0.200), relief, caring cải thiện → few-shot giúp một phần nhưng bất ổn với lớp cực hiếm.

> k=5 ví dụ lấy ngẫu nhiên từ training (seed 42, tách rời test). Parse phản hồi qua nhiều fallback: parse full JSON → trích {…} rồi […] → cuối cùng fallback ["neutral"] tránh vector rỗng.

## EXP-07 · LLM Ensemble (lấy cảm hứng từ PAI)
**Script:** `python -m scripts.ensemble_llm`
- **Setup:** ghép prediction-set của Llama/Qwen × zero/few-shot trên cùng 5,427 mẫu bằng voting; gần như zero-cost (chỉ post-process JSON, không GPU).

| Strategy | F1-macro | F1-micro | Hamming |
|----------|----------|----------|---------|
| Qwen FS (best single) | 0.2466 | 0.2900 | 0.0710 |
| FS intersection (Llama∩Qwen) | 0.2475 | 0.3566 | 0.0543 |
| **All-4 majority (≥2)** | **0.2657** | 0.2848 | 0.0938 |
| All-4 majority (≥3) | 0.2592 | 0.3564 | 0.0552 |

- **Kết quả:** majority (≥2) = **0.2657** (+7.7% so với LLM đơn tốt nhất) → xác nhận hướng của PAI (ghép nhiều LLM tốt hơn từng cái). Nhưng vẫn thua BERT t=0.5 (0.4148) ~1.56×. Bài học: ensemble prompt-only **giúp nhưng không thay** fine-tuning; PAI đạt SOTA nhờ ensemble các LLM **đã LoRA fine-tune**.

## EXP-08 · Cross-benchmark trên BRIGHTER (SemEval-2025 Task 11)
**Script:** `python -m scripts.run_brighter --mode bert|llm --model ... [--prompt ...]`
- **Dữ liệu:** `brighter-dataset/BRIGHTER-emotion-categories`, config `eng` (train 2,764 / dev 230 / test 5,528; 5 lớp, English không annotate disgust).
- **Kết quả then chốt:** BERT-base = **0.7069 ≈ baseline chính thức 0.708** (Δ 0.001) → xác nhận pipeline fine-tune đúng chuẩn; điểm GoEmotions thấp (0.41) chỉ do 28 lớp khó hơn 5 lớp, **không** phải method yếu.
- **Granularity quyết định khoảng cách:** trên BRIGHTER (5 lớp) LLM prompt-only đạt tới 84% của BERT (0.597 vs 0.707); trên GoEmotions (28 lớp) chỉ 59% (0.247 vs 0.415).

## EXP-09 · LoRA Fine-tune Qwen2.5-3B (đóng vòng lập luận)
**Script:** `python -m scripts.lora_finetune --dataset goemotions|brighter`
- **Setup:** Qwen2.5-3B-Instruct làm `AutoModelForSequenceClassification` + LoRA (peft 0.19.1, r=16, target q/k/v/o_proj, **bf16** — fp16 lỗi GradScaler trên RTX Ada).

| Same Qwen2.5-3B | GoEmotions (28 lớp) | BRIGHTER (5 lớp) |
|-----------------|---------------------|-------------------|
| Zero-shot (prompt) | 0.2364 | 0.4662 |
| Few-shot (prompt) | 0.2466 | 0.5966 |
| **LoRA fine-tune** | **0.4519** | **0.7522** |
| → vs BERT-base (t=0.5) | 0.4148 (LoRA thắng) | 0.7069 (LoRA thắng) |

- **Kết luận:** cùng một Qwen-3B, prompt→LoRA nhảy **+0.21 / +0.29**, vượt BERT-base trên cả hai dataset, đạt **91% điểm PAI (0.823)**. Khoảng cách còn lại tới PAI = quy mô lớn hơn (32B) + ensemble stacking 2 vòng. Bài học không phải "LLM yếu" mà "**LLM phải được fine-tune**".

---

## Phân tích bổ sung

**Error analysis BERT vs Qwen (full 5,427):**

| Category | Count | % |
|----------|-------|---|
| BERT closer (Jaccard cao hơn) | 2,201 | 40.6% |
| Both wrong (tie) | 1,735 | 32.0% |
| LLM wins (exact match) | 515 | 9.5% |
| BERT wins (exact match) | 457 | 8.4% |
| LLM closer | 270 | 5.0% |
| Both correct | 248 | 4.6% |
| **Disagreement rate** | **5,066 / 5,427** | **93.3%** |

> Lỗi phổ biến nhất của LLM: gán cảm xúc cho text trung tính (vd. "KAMALA 2020!!!!!!" → Qwen đoán excitement). Ngược lại khi 1 cảm xúc rõ, LLM tránh over-label của BERT (vd. "Fuck you." → BERT thêm annoyance, Qwen chỉ trả anger). Tổng thể: BERT mạnh ở partial overlap (ít miss nhãn chính), LLM chính xác hơn ở cảm xúc đơn rõ ràng nhưng yếu recall và hay sai ở neutral/lớp hiếm.

**Cost efficiency (RQ4):**

| Model | Training | Inference |
|-------|----------|-----------|
| BERT fine-tune | ~37 phút (1 lần) | <1 phút (cả test); ~0.01 s/mẫu (batched) |
| Qwen2.5-3B (prompt) | — | ~0.48 s/mẫu |
| Llama-3.2-3B (prompt) | — | ~1.33 s/mẫu |

> Ở quy mô production, BERT xong dưới 10 phút trong khi prompt-only LLM mất hàng giờ và kém chính xác hơn. Threshold tuning (không tốn thêm compute) cộng thêm +0.10 F1-macro — cải thiện "miễn phí".

---

## Trả lời câu hỏi nghiên cứu

| RQ | Trả lời |
|----|---------|
| RQ1 | Fine-tuned ≫ prompt-only LLM (~1.7–2×) trên GoEmotions — **xác nhận mạnh**. |
| RQ2 | BERT ≈ RoBERTa qua 3 epoch (gap 0.0037; cần multi-seed để kết luận). |
| RQ3 | LLM **KHÔNG** mạnh hơn ở lớp hiếm — **bác bỏ** giả thuyết. |
| RQ4 | Fine-tuning thắng về chi phí: train 1 lần, infer rất nhanh. |

---

## Phạm vi & lưu ý
- LLM API thương mại (vd. Gemini) **ngoài phạm vi** — chỉ dùng LLM mã nguồn mở chạy local để bảo đảm tái lập và công bằng điều kiện (mô hình mở, không phụ thuộc billing, kiểm soát được cấu hình).
- Model weights (`results/models/`) và predictions thô (`results/llm/`) **không** đưa lên repo (lớn, sinh lại được); chỉ commit `results/metrics/*.json`.
- Toàn bộ thí nghiệm dùng seed = 42 (random/numpy/torch/cuda, cudnn.deterministic = True); LLM dùng temperature = 0.
