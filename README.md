# GoEmotions Emotion Classification: BERT Fine-tuning vs LLM In-context Learning

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/pytorch-2.0+-red.svg" />
  <img src="https://img.shields.io/badge/transformers-4.30+-yellow.svg" />
  <img src="https://img.shields.io/badge/license-MIT-green.svg" />
  <img src="https://img.shields.io/badge/status-completed-brightgreen.svg" />
</p>

> Course project — **Natural Language Processing**
> Master of Computer Science, Vietnam Japan University (VJU-VNU)
> May 2026

---

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Câu hỏi nghiên cứu](#-câu-hỏi-nghiên-cứu)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Kết quả chính](#-kết-quả-chính)
- [Cài đặt](#-cài-đặt)
- [Cách sử dụng](#-cách-sử-dụng)
- [Cấu trúc repo](#-cấu-trúc-repo)
- [Roadmap](#-roadmap)
- [Tham khảo](#-tham-khảo)
- [License](#-license)

---

## 🎯 Tổng quan

Project này so sánh hai phương pháp tiếp cận **multi-label emotion classification** trên bộ dữ liệu [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) (Google Research, 2020):

| Phương pháp | Mô hình | Kỹ thuật |
|-------------|---------|----------|
| **Fine-tuning** | BERT-base, RoBERTa-base | Supervised learning trên 43K training samples |
| **In-context Learning** | Llama 3.2 3B, Qwen2.5 3B (mã nguồn mở, chạy local) | Zero-shot và Few-shot prompting (k=5) |
| **LLM Ensemble** | Llama + Qwen (voting) | Mô phỏng hệ đoạt giải PAI tại SemEval-2025 Task 11 |

> **Phạm vi:** so sánh fine-tuned encoder vs **LLM mã nguồn mở chạy local**. LLM API thương mại (Gemini, GPT…) nằm ngoài phạm vi (mô hình đóng, tính phí theo lượt gọi, không kiểm soát được cấu hình).

**Dataset:** 58K Reddit comments được gán nhãn với 27 emotions + neutral, đặc trưng bởi:
- Multi-label (17% samples có ≥2 emotions)
- Class imbalance nặng (ratio 184x giữa class phổ biến nhất và hiếm nhất)
- Text ngắn (median 12 từ)

---

## ❓ Câu hỏi nghiên cứu

1. **RQ1:** Fine-tuned BERT có vượt qua LLM zero-shot trong fine-grained emotion classification (28 classes) không?
2. **RQ2:** Few-shot prompting có thu hẹp được gap giữa LLM và fine-tuned model không?
3. **RQ3:** Trên các rare classes (grief, pride, relief), LLM có lợi thế gì so với fine-tuning?
4. **RQ4:** Trade-off giữa accuracy, inference cost, và training cost của hai phương pháp như thế nào?

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                    GoEmotions Dataset (58K)                     │
│              43K train / 5K val / 5K test                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┴────────────────┐
            ▼                                ▼
    ┌──────────────────┐            ┌────────────────────┐
    │  Track A: BERT   │            │  Track B: LLM      │
    │  Fine-tuning     │            │  local (open-src)  │
    ├──────────────────┤            ├────────────────────┤
    │ • BERT-base      │            │ • Llama 3.2 3B     │
    │ • RoBERTa-base   │            │ • Qwen2.5 3B       │
    │ • BCEWithLogits  │            │ • Zero/Few-shot    │
    │   + pos_weight   │            │ • Ensemble (vote)  │
    │ • 3 epochs       │            │                    │
    └──────────────────┘            └────────────────────┘
            │                                │
            └───────────────┬────────────────┘
                            ▼
            ┌────────────────────────────────┐
            │   Evaluation & Analysis        │
            │                                │
            │  • F1-macro / F1-micro         │
            │  • Per-class F1                │
            │  • Confusion analysis          │
            │  • Disagreement analysis       │
            │  • Cost-benefit analysis       │
            └────────────────────────────────┘
```

---

## 📊 Kết quả chính

> ✅ **Hoàn tất** — tất cả LLM đánh giá trên **full test 5,427 mẫu** (cùng tập với BERT, so sánh công bằng).

| Model | Method | F1-macro | F1-micro | Hamming Loss |
|-------|--------|----------|----------|--------------|
| **BERT-base** | Fine-tune, **t=0.9 (tuned)** | **0.5167** | 0.5278 | — |
| BERT-base | Fine-tune, t=0.5 | 0.4148 ±0.0008 (3 seeds) | 0.4660 | 0.0778 |
| RoBERTa-base | Fine-tune, t=0.5 | 0.4111 | 0.4618 | 0.0795 |
| LLM Ensemble | majority ≥2 (EXP-07) | 0.2657 | 0.2848 | 0.0938 |
| Qwen2.5 3B | Few-shot (k=5) | 0.2466 | 0.2900 | 0.0710 |
| Llama 3.2 3B | Few-shot (k=5) | 0.2382 | 0.2432 | 0.1085 |
| Qwen2.5 3B | Zero-shot | 0.2364 | 0.2703 | 0.0713 |
| Llama 3.2 3B | Zero-shot | 0.2133 | 0.2329 | 0.1047 |

**Phát hiện chính:**
- **Fine-tuned BERT vượt mọi LLM mã nguồn mở** (zero/few-shot, kể cả ensemble) — RQ1 confirmed mạnh (~1.7–2×).
- **Threshold tuning** (t=0.5→0.9, không tốn compute) đưa BERT lên **0.5167**, vượt paper baseline (0.46).
- **Ensemble LLM** nâng F1-macro +7.7% (đúng hướng PAI) nhưng vẫn chưa đuổi kịp BERT → fine-tuning mới là chìa khoá lên SOTA.
- **Kiểm chứng chéo (EXP-08):** chạy chính pipeline trên dữ liệu **SemEval-2025 Task 11 (BRIGHTER English, 5 lớp)**, BERT-base đạt **F1-macro 0.7069 ≈ baseline chính thức 0.708** → xác nhận pipeline đúng chuẩn; điểm GoEmotions thấp chỉ vì **28 lớp khó hơn**. LLM off-the-shelf đạt 84% của BERT ở 5 lớp (vs 59% ở 28 lớp).
- **LoRA fine-tune (EXP-09):** fine-tune chính **Qwen2.5-3B** đạt **0.4519 (GoEmotions) / 0.7522 (BRIGHTER)** — cùng model nhảy +0.21/+0.29 so với prompt, **vượt BERT-base ở cả hai dataset** → chứng minh tự thân *fine-tuning mới là chìa khoá*.

**Baseline để so sánh:** Paper gốc Demszky et al. (2020) báo cáo BERT-base F1-macro = **0.46**.

---

## 🔧 Cài đặt

### Yêu cầu hệ thống
- Python 3.10+
- CUDA 12.x (cho GPU training)
- 4GB+ VRAM (khuyến nghị 16GB+ cho batch size lớn)

### Setup môi trường

```bash
# Clone repo
git clone https://github.com/<your-username>/goemotions-emotion-classification.git
cd goemotions-emotion-classification

# Tạo conda environment
conda create -n goemotions python=3.10 -y
conda activate goemotions

# Cài PyTorch với CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Cài dependencies
pip install -r requirements.txt
```

### Setup (không cần API key cho LLM track)

LLM track chạy **local qua HuggingFace** (Llama/Qwen) nên **không cần API key**. Chỉ cần đăng nhập HuggingFace để tải model (nếu repo gated) và (tuỳ chọn) W&B để log training:

```bash
# (tuỳ chọn) Tạo file .env cho W&B logging
cp .env.example .env
# WANDB_API_KEY=your_key_here   (lấy từ https://wandb.ai/authorize)

# (nếu cần) đăng nhập HuggingFace để tải Llama/Qwen
huggingface-cli login
```

---

## 🚀 Cách sử dụng

### 1. Exploratory Data Analysis

```bash
python scripts/run_eda.py
```

→ Output: phân phối nhãn, độ dài text, thống kê multi-label được lưu vào `results/eda/`.

### 2. Train BERT baseline

```bash
# Test pipeline với debug mode (1000 samples, 1 epoch)
python -m src.train --config configs/bert_base_debug.yaml

# Train full
python -m src.train --config configs/bert_base.yaml
```

### 3. Train RoBERTa để so sánh

```bash
python -m src.train --config configs/roberta_base.yaml
```

### 4. Chạy LLM inference (local, full 5,427 test)

```bash
# Zero-shot
python -m src.llm_inference --config configs/qwen_zeroshot.yaml
python -m src.llm_inference --config configs/llama_zeroshot.yaml

# Few-shot (k=5)
python -m src.llm_inference --config configs/qwen_fewshot.yaml
python -m src.llm_inference --config configs/llama_fewshot.yaml

# LLM Ensemble (EXP-07) — ghép predictions Llama+Qwen bằng voting
python -m scripts.ensemble_llm
```

### 5. Phân tích kết quả

```bash
# Tổng hợp metrics, vẽ plots
python scripts/compare_results.py

# Error analysis BERT vs LLM (full 5,427)
python scripts/error_analysis.py --llm results/llm/qwen_zeroshot/predictions.json
```

---

## 📁 Cấu trúc repo

```
goemotions-emotion-classification/
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example                 # Template cho API keys
├── .gitignore
│
├── src/                         # Core source code
│   ├── __init__.py
│   ├── data.py                  # Dataset, DataLoader, preprocessing
│   ├── models.py                # EmotionClassifier (BERT/RoBERTa)
│   ├── train.py                 # Training loop
│   ├── evaluate.py              # Evaluation metrics
│   ├── llm_inference.py         # LLM inference (Llama/Qwen local; Gemini client out-of-scope)
│   ├── prompts.py               # Prompt templates (zero/few-shot)
│   └── utils.py                 # Helpers (seed, logging, etc.)
│
├── configs/                     # YAML configs cho experiments
│   ├── bert_base.yaml
│   ├── roberta_base.yaml
│   ├── llama_zeroshot.yaml      # + llama_fewshot.yaml
│   └── qwen_zeroshot.yaml       # + qwen_fewshot.yaml
│
├── scripts/                     # Standalone scripts
│   ├── run_eda.py               # EDA
│   ├── run_multiseed.py         # BERT multi-seed
│   ├── tune_threshold.py        # Threshold tuning
│   ├── ensemble_llm.py          # EXP-07 LLM ensemble
│   ├── error_analysis.py        # BERT vs LLM disagreement
│   └── compare_results.py       # Tổng hợp + plots
│
├── notebooks/                   # Jupyter notebooks (exploration, analysis)
│   ├── 01_eda.ipynb
│   ├── 02_baseline_results.ipynb
│   └── 03_error_analysis.ipynb
│
├── tests/                       # Unit tests
│   ├── test_data.py
│   └── test_metrics.py
│
├── docs/                        # Documentation
│   ├── PROJECT_PLAN.md          # Kế hoạch chi tiết
│   ├── METHODOLOGY.md           # Phương pháp luận
│   ├── EXPERIMENTS.md           # Log các experiments
│   └── REPORT.md                # Báo cáo cuối kỳ
│
├── data/                        # Dataset (gitignored)
│   ├── raw/
│   └── processed/
│
├── results/                     # Outputs (selective gitignore)
│   ├── models/                  # Model checkpoints (gitignored)
│   ├── plots/                   # Visualizations
│   ├── logs/                    # Training logs
│   └── metrics/                 # JSON metrics
│
└── .github/
    └── workflows/
        └── lint.yml             # CI: code style check
```

---

## 🗺️ Roadmap

Xem chi tiết tại [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md).

### Milestone 1: Foundation — ✅ Done
- [x] EDA & dataset analysis
- [x] Repo structure setup
- [x] BERT baseline training script
- [x] Evaluation pipeline

### Milestone 2: Experiments — ✅ Done
- [x] BERT-base full training (+ multi-seed, threshold tuning)
- [x] RoBERTa-base comparison
- [x] Llama/Qwen zero-shot inference (full 5,427)
- [x] Llama/Qwen few-shot inference (k=5)
- [x] LLM Ensemble (EXP-07, mô phỏng PAI)

### Milestone 3: Analysis & Writeup — ✅ Done
- [x] Error analysis (BERT vs LLM, full 5,427)
- [x] Disagreement analysis
- [x] So sánh & định vị với SemEval-2025 Task 11
- [x] Final report (tiếng Việt) + slides

### Milestone 4: Polish & Submit — ✅ Done
- [x] Code review & refactoring
- [x] Documentation polish
- [x] Final submission

---

## 📚 Tham khảo

### Papers chính

1. **Demszky, D. et al. (2020).** *GoEmotions: A Dataset of Fine-Grained Emotions.* ACL 2020. [[paper](https://arxiv.org/abs/2005.00547)] [[dataset](https://github.com/google-research/google-research/tree/master/goemotions)]

2. **Devlin, J. et al. (2019).** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019. [[paper](https://arxiv.org/abs/1810.04805)]

3. **Liu, Y. et al. (2019).** *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv. [[paper](https://arxiv.org/abs/1907.11692)]

4. **Brown, T. et al. (2020).** *Language Models are Few-Shot Learners.* NeurIPS 2020. [[paper](https://arxiv.org/abs/2005.14165)]

5. **Muhammad, S. H. et al. (2025).** *SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection.* [[paper](https://arxiv.org/abs/2503.07269)]

6. **Ruan, Z. et al. (2025).** *PAI at SemEval-2025 Task 11: A Large Language Model Ensemble Strategy.* [[paper](https://aclanthology.org/2025.semeval-1.150/)]

### Tools & Libraries

- [HuggingFace Transformers](https://huggingface.co/transformers/) — Llama 3.2, Qwen2.5
- [PyTorch](https://pytorch.org/)
- [Weights & Biases](https://wandb.ai/)

---

## 📝 License

MIT License — xem [LICENSE](LICENSE) để biết chi tiết.

---

## 🙏 Acknowledgments

- **Google Research** — GoEmotions dataset
- **HuggingFace** — Pre-trained models và libraries

---

<p align="center">
  <i>Made with ☕ at VJU-VNU, Hanoi, Vietnam</i>
</p>
