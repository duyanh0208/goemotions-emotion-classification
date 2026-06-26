# 🧪 Experiments Log

Đầy đủ 9 thí nghiệm (EXP-01 → EXP-09) đã hoàn tất. Tất cả số liệu là **F1-macro**, đánh giá trên **full test** (GoEmotions: 5,427 mẫu; BRIGHTER English: 5,528 mẫu). Chi tiết phân tích xem [REPORT.md](REPORT.md).

| Phần cứng | NVIDIA RTX 2000 Ada 16GB · CUDA 12.x · seed = 42 |
|---|---|

---

## Bảng tổng hợp (GoEmotions, 28 lớp)

| # | Experiment | Phương pháp | F1-macro | F1-micro | Ghi chú |
|---|-----------|-------------|----------|----------|---------|
| 01 | BERT-base | Fine-tune (t=0.5) | **0.4148 ±0.0008** | 0.4660 | 3 seeds (42/123/456) |
| 01 | BERT-base | Fine-tune (t=0.9 tuned) | **0.5167** | 0.5278 | vượt baseline 0.46 |
| 02 | RoBERTa-base | Fine-tune (t=0.5) | 0.4111 | 0.4618 | t=0.9 → 0.5136 |
| 03 | Llama-3.2-3B | Zero-shot | 0.2133 | 0.2329 | ~1.33 s/mẫu |
| 04 | Qwen2.5-3B | Zero-shot | 0.2364 | 0.2703 | ~0.48 s/mẫu |
| 05 | Llama-3.2-3B | Few-shot (k=5) | 0.2382 | 0.2432 | +11.7% vs ZS |
| 06 | Qwen2.5-3B | Few-shot (k=5) | 0.2466 | 0.2900 | +4.3% vs ZS |
| 07 | LLM Ensemble | majority ≥2 (4 sets) | 0.2657 | 0.2848 | +7.7% vs LLM đơn |
| 09 | Qwen2.5-3B | **LoRA fine-tune** | **0.4519** | 0.5928 | vượt BERT-base |
| — | *Paper baseline* | *BERT* | *0.46* | — | Demszky 2020 |

## Kiểm chứng chéo — SemEval-2025 Task 11 (BRIGHTER English, 5 lớp)

| # | Experiment | Phương pháp | F1-macro | Ghi chú |
|---|-----------|-------------|----------|---------|
| 08 | BERT-base | Fine-tune | **0.7069** | ≈ baseline chính thức 0.708 |
| 08 | Qwen2.5-3B | Zero-shot | 0.4662 | — |
| 08 | Qwen2.5-3B | Few-shot | 0.5966 | — |
| 08 | Llama-3.2-3B | Zero-shot | 0.5700 | — |
| 08 | Llama-3.2-3B | Few-shot | 0.5778 | — |
| 09 | Qwen2.5-3B | **LoRA fine-tune** | **0.7522** | vượt BERT & baseline |
| — | *RoBERTa baseline* | *fine-tune* | *0.708* | official |
| — | *PAI (đoạt giải)* | *ensemble + AdaLoRA + stacking* | *0.823* | SOTA |

---

## EXP-01 · BERT-base Fine-tuning
**Config:** `configs/bert_base.yaml` · **Script:** `python -m src.train --config configs/bert_base.yaml`
- **Setup:** bert-base-uncased, BCEWithLogitsLoss + pos_weight (clip [1,50]), lr 2e-5, batch 16, 3 epochs, AdamW. Thời gian ~37 phút/seed.
- **Kết quả (test 5,427):** F1-macro 0.4159 (seed 42); multi-seed (`scripts/run_multiseed.py`) = **0.4148 ± 0.0008**.
- **Threshold tuning** (`scripts/tune_threshold.py`): t=0.9 → **F1-macro 0.5167** (+0.10), vượt paper baseline 0.46. Lớp hiếm hưởng lợi nhiều nhất (grief 0.222→0.400, relief 0.193→0.400).
- **Observations:** Val Loss tăng nhẹ ở epoch 3 (hơi overfit) trong khi Val F1 vẫn cải thiện.

## EXP-02 · RoBERTa-base Fine-tuning
**Config:** `configs/roberta_base.yaml`
- **Kết quả:** F1-macro **0.4111** (t=0.5), **0.5136** (t=0.9). ~38 phút.
- **Observations:** đường cong kém ổn định hơn BERT (Val F1 tụt ở epoch 2). Chênh BERT vs RoBERTa 0.0037 < ý nghĩa thống kê (cần multi-seed RoBERTa).

## EXP-03 · Llama-3.2-3B Zero-shot
**Config:** `configs/llama_zeroshot.yaml` · **Script:** `python -m src.llm_inference --config ...`
- **Setup:** meta-llama/Llama-3.2-3B-Instruct, local, float16, greedy (do_sample=False), prompt JSON 28 nhãn.
- **Kết quả:** F1-macro **0.2133**. Tốc độ ~1.33 s/mẫu. Profile "aggressive" (Hamming ~0.105), bắt lớp hiếm tốt hơn Qwen.

## EXP-04 · Qwen2.5-3B Zero-shot
**Config:** `configs/qwen_zeroshot.yaml`
- **Kết quả:** F1-macro **0.2364**. ~0.48 s/mẫu. Profile "conservative" (Hamming ~0.071), miss grief & caring (=0.000).
- **RQ1 xác nhận:** BERT > Qwen ZS gần 2×. **RQ3 bác bỏ:** LLM không mạnh hơn ở lớp hiếm.

## EXP-05 · Llama-3.2-3B Few-shot (k=5)
**Config:** `configs/llama_fewshot.yaml`
- **Kết quả:** F1-macro **0.2382** (+11.7% so với zero-shot). Few-shot loại bỏ các lớp F1=0.000.

## EXP-06 · Qwen2.5-3B Few-shot (k=5)
**Config:** `configs/qwen_fewshot.yaml`
- **Kết quả:** F1-macro **0.2466** (+4.3%). Vẫn thua BERT (t=0.5) ~1.68×.

## EXP-07 · LLM Ensemble
**Script:** `python -m scripts.ensemble_llm`
- **Setup:** ghép 4 prediction-set (Llama/Qwen × zero/few-shot) bằng voting; không cần GPU.
- **Kết quả:** majority (≥2) = **0.2657** (+7.7% so với LLM đơn tốt nhất). Vẫn thua BERT — ensemble prompt-only không thay được fine-tuning.

## EXP-08 · Cross-benchmark trên BRIGHTER (SemEval-2025 Task 11)
**Script:** `python -m scripts.run_brighter --mode bert|llm --model ... [--prompt ...]`
- **Dữ liệu:** `brighter-dataset/BRIGHTER-emotion-categories`, config `eng` (5 lớp; English không annotate disgust). Test 5,528.
- **Kết quả then chốt:** BERT-base = **0.7069 ≈ baseline chính thức 0.708** → xác nhận pipeline đúng chuẩn; điểm GoEmotions thấp (0.41) chỉ do 28 lớp khó hơn 5 lớp. LLM prompt: Qwen ZS 0.4662 / FS 0.5966; Llama ZS 0.5700 / FS 0.5778.

## EXP-09 · LoRA Fine-tune Qwen2.5-3B (đóng vòng lập luận)
**Script:** `python -m scripts.lora_finetune --dataset goemotions|brighter`
- **Setup:** Qwen2.5-3B-Instruct làm `AutoModelForSequenceClassification` + LoRA (peft 0.19.1, r=16, target q/k/v/o_proj, **bf16** — fp16 lỗi GradScaler trên RTX Ada).
- **Kết quả:** GoEmotions **0.4519** (vượt BERT-base 0.4148); BRIGHTER **0.7522** (vượt BERT 0.7069 và baseline 0.708, đạt 91% điểm PAI 0.823).
- **Kết luận:** cùng một Qwen-3B, prompt→LoRA nhảy +0.21/+0.29 → *yếu tố quyết định là fine-tuning, không phải họ model*.

---

## Phạm vi & lưu ý
- LLM API thương mại (vd. Gemini) **ngoài phạm vi** — chỉ dùng LLM mã nguồn mở chạy local để bảo đảm tái lập và công bằng điều kiện.
- Model weights (`results/models/`) và predictions thô (`results/llm/`) **không** đưa lên repo (lớn, sinh lại được); chỉ commit `results/metrics/*.json`.
