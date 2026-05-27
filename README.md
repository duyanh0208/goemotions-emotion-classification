# GoEmotions Emotion Classification: BERT Fine-tuning vs LLM In-context Learning

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/pytorch-2.0+-red.svg" />
  <img src="https://img.shields.io/badge/transformers-4.30+-yellow.svg" />
  <img src="https://img.shields.io/badge/license-MIT-green.svg" />
  <img src="https://img.shields.io/badge/status-in--progress-orange.svg" />
</p>

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
| **In-context Learning** | Gemini 2.0 Flash | Zero-shot và Few-shot prompting (5 examples) |

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
    │  Fine-tuning     │            │  In-context        │
    ├──────────────────┤            ├────────────────────┤
    │ • BERT-base      │            │ • Gemini Flash     │
    │ • RoBERTa-base   │            │ • Zero-shot        │
    │ • BCEWithLogits  │            │ • Few-shot (k=5)   │
    │   + pos_weight   │            │ • JSON output mode │
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

> **🚧 Đang cập nhật** — Kết quả sẽ được điền sau khi hoàn thành các experiments.

| Model | Method | F1-macro | F1-micro | Hamming Loss |
|-------|--------|----------|----------|--------------|
| BERT-base | Fine-tune | TBD | TBD | TBD |
| RoBERTa-base | Fine-tune | TBD | TBD | TBD |
| Gemini Flash | Zero-shot | TBD | TBD | TBD |
| Gemini Flash | Few-shot (k=5) | TBD | TBD | TBD |

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

### Setup API keys

```bash
# Tạo file .env (không commit lên Git!)
cp .env.example .env
# Mở .env và điền:
# GEMINI_API_KEY=your_key_here  (lấy từ https://aistudio.google.com/apikey)
# WANDB_API_KEY=your_key_here   (lấy từ https://wandb.ai/authorize)
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

### 4. Chạy LLM inference

```bash
# Zero-shot
python -m src.llm_inference --config configs/gemini_zeroshot.yaml --n_samples 2000

# Few-shot
python -m src.llm_inference --config configs/gemini_fewshot.yaml --n_samples 2000
```

### 5. Phân tích kết quả

```bash
# Tổng hợp metrics, vẽ plots
python scripts/compare_results.py

# Error analysis chi tiết
jupyter notebook notebooks/03_error_analysis.ipynb
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
│   ├── llm_inference.py         # Gemini API client
│   ├── prompts.py               # Prompt templates
│   └── utils.py                 # Helpers (seed, logging, etc.)
│
├── configs/                     # YAML configs cho experiments
│   ├── bert_base.yaml
│   ├── bert_base_debug.yaml
│   ├── roberta_base.yaml
│   ├── gemini_zeroshot.yaml
│   └── gemini_fewshot.yaml
│
├── scripts/                     # Standalone scripts
│   ├── run_eda.py
│   ├── compare_results.py
│   └── generate_report_plots.py
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

### Milestone 1: Foundation (Week 1) — ✅ In Progress
- [x] EDA & dataset analysis
- [x] Repo structure setup
- [ ] BERT baseline training script
- [ ] Evaluation pipeline

### Milestone 2: Experiments (Week 2)
- [ ] BERT-base full training
- [ ] RoBERTa-base comparison
- [ ] Gemini zero-shot inference
- [ ] Gemini few-shot inference

### Milestone 3: Analysis & Writeup (Week 3)
- [ ] Error analysis
- [ ] Disagreement analysis (BERT vs LLM)
- [ ] Final report (8-10 trang tiếng Việt)
- [ ] Presentation slides

### Milestone 4: Polish & Submit (Week 4)
- [ ] Code review & refactoring
- [ ] Documentation polish
- [ ] Final submission

---

## 📚 Tham khảo

### Papers chính

1. **Demszky, D. et al. (2020).** *GoEmotions: A Dataset of Fine-Grained Emotions.* ACL 2020. [[paper](https://arxiv.org/abs/2005.00547)] [[dataset](https://github.com/google-research/google-research/tree/master/goemotions)]

2. **Devlin, J. et al. (2019).** *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019. [[paper](https://arxiv.org/abs/1810.04805)]

3. **Liu, Y. et al. (2019).** *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv. [[paper](https://arxiv.org/abs/1907.11692)]

4. **Brown, T. et al. (2020).** *Language Models are Few-Shot Learners.* NeurIPS 2020. [[paper](https://arxiv.org/abs/2005.14165)]

### Tools & Libraries

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [Weights & Biases](https://wandb.ai/)
- [Google Gemini API](https://ai.google.dev/)

---

## 📝 License

MIT License — xem [LICENSE](LICENSE) để biết chi tiết.

---

## 🙏 Acknowledgments

- **PGS.TS. Trần Thị Oanh** — Giảng viên môn NLP, IS-VNU
- **Google Research** — GoEmotions dataset
- **HuggingFace** — Pre-trained models và libraries

---

<p align="center">
  <i>Made with ☕ at VJU-VNU, Hanoi, Vietnam</i>
</p>
