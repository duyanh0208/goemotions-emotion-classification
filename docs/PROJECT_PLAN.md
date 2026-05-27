# 📋 Project Plan — GoEmotions Emotion Classification

> **Document version:** 1.0 (May 2026)
> **Author:** Bùi Đào Duy Anh
> **Status:** In Progress

---

## 1. Bối cảnh và Động lực

### 1.1. Bài toán

Emotion classification là một task quan trọng trong NLP với nhiều ứng dụng thực tế: customer feedback analysis, mental health monitoring, content moderation, conversational AI. Tuy nhiên, đây là bài toán **khó** vì:

- **Ambiguity:** Cùng một câu có thể mang nhiều cảm xúc (multi-label)
- **Context-dependent:** Cảm xúc phụ thuộc ngữ cảnh, sarcasm, văn hóa
- **Imbalance:** Các cảm xúc hiếm (grief, pride) khó train hơn
- **Granularity:** Số class lớn (28 vs truyền thống 6-7 Ekman emotions)

### 1.2. Vì sao chọn GoEmotions?

- **Quy mô lớn:** 58K samples, đủ để fine-tune deep models
- **Granularity cao:** 27 emotions + neutral (chi tiết hơn Ekman)
- **Quality cao:** Annotated bởi Google, multiple annotators per sample
- **Multi-label:** Phản ánh thực tế (1 text có thể có nhiều emotions)
- **Benchmark có sẵn:** F1-macro = 0.46 từ paper gốc → có target để so sánh

### 1.3. Vì sao so sánh BERT vs LLM?

Đây là câu hỏi **thực tiễn quan trọng** năm 2024-2026:
- Doanh nghiệp nên đầu tư fine-tune model hay dùng LLM API?
- LLM zero-shot có "đủ tốt" cho fine-grained tasks không?
- Few-shot learning có thay thế được fine-tuning?

Project này cung cấp empirical evidence cho các câu hỏi đó.

---

## 2. Câu hỏi nghiên cứu

| # | Câu hỏi | Cách trả lời |
|---|---------|--------------|
| RQ1 | Fine-tuned BERT có vượt LLM zero-shot? | So sánh F1-macro |
| RQ2 | Few-shot có thu hẹp gap? | Compare zero-shot vs few-shot LLM |
| RQ3 | LLM mạnh ở rare classes? | Per-class F1 analysis |
| RQ4 | Trade-off accuracy vs cost? | Pareto analysis |

---

## 3. Phương pháp

### 3.1. Dataset

**GoEmotions** (Demszky et al., 2020):
- Source: Reddit comments (2005-2019)
- Splits: 43,410 train / 5,426 val / 5,427 test
- Labels: 27 emotions + neutral (28 classes)
- Format: Multi-label (mỗi text có 1-5 emotions)

**Preprocessing:**
- Multi-hot encoding cho labels (28-dim binary vector)
- Tokenization với BERT/RoBERTa tokenizer
- Max length: 128 tokens (đủ cho 99.97% samples)

### 3.2. Track A: Fine-tuning

**Models:**
- `bert-base-uncased` (110M params)
- `roberta-base` (125M params)

**Architecture:**
```
Input text
  → Tokenizer
  → BERT/RoBERTa encoder
  → [CLS] embedding (768-dim)
  → Dropout (p=0.1)
  → Linear(768, 28)
  → Sigmoid (multi-label)
```

**Training:**
- Loss: `BCEWithLogitsLoss` với `pos_weight` (xử lý imbalance)
- Optimizer: AdamW, lr=2e-5, weight_decay=0.01
- Scheduler: Linear warmup (10%) → Linear decay
- Batch size: 16-32 (tùy GPU)
- Epochs: 3
- Gradient clipping: max_norm=1.0

**Class imbalance handling:**
- `pos_weight[c] = (N_negative_c) / (N_positive_c)`, clip ở [1, 50]
- Rationale: Class hiếm → weight cao → model bị "phạt" nặng hơn khi miss

### 3.3. Track B: LLM In-context Learning

**Model:** Gemini 2.0 Flash (Google AI)
- Free tier: 1500 requests/day
- JSON mode native (structured output)

**Zero-shot prompt:**
```
You are an emotion classifier. Given a text, identify which of the
following 28 emotions apply (multi-label):
[list 28 emotions with brief descriptions]

Text: "{input}"
Output a JSON list of applicable emotion names.
```

**Few-shot prompt (k=5):**
- Cùng base prompt + 5 examples đa dạng từ training set
- Examples chọn để cover các emotion types (positive, negative, neutral)

**Evaluation:** Cùng test set như Track A (subset 2000 samples vì quota)

### 3.4. Metrics

| Metric | Tại sao dùng |
|--------|--------------|
| **F1-macro** (primary) | Treats all classes equally → fair với imbalanced data |
| F1-micro | Reflects overall performance |
| F1-weighted | Accounts for class frequency |
| Hamming Loss | Multi-label specific |
| Per-class F1 | Để phân tích chi tiết từng emotion |

### 3.5. Analysis

1. **Aggregate comparison:** Bảng F1 cho 4 setups
2. **Per-class breakdown:** Heatmap F1 cho 28 classes × 4 models
3. **Disagreement analysis:** 50 samples BERT vs LLM khác kết quả
4. **Cost analysis:** GPU-hours (BERT) vs API tokens (LLM)
5. **Error categorization:** ambiguity, sarcasm, multi-emotion confusion

---

## 4. Timeline

### Week 1 (20-26 May 2026): Foundation
| Day | Task | Deliverable |
|-----|------|-------------|
| Tue | ✅ Setup repo, EDA | EDA notebook, label distribution |
| Wed | Implement `data.py`, `models.py` | Core modules |
| Thu | Implement `train.py` | Training loop |
| Fri | Train BERT debug + full | Baseline metrics |
| Sat | Implement `evaluate.py` | Per-class F1, plots |
| Sun | Train RoBERTa | RoBERTa metrics |

### Week 2 (27 May - 2 June): LLM Track + Analysis
| Day | Task | Deliverable |
|-----|------|-------------|
| Mon | Setup Gemini API, `llm_inference.py` | Working LLM client |
| Tue | Run zero-shot on 500 samples (smoke test) | Initial results |
| Wed | Run zero-shot on 2000 test samples | Full zero-shot metrics |
| Thu | Run few-shot on 2000 test samples | Few-shot metrics |
| Fri | Error analysis notebook | 03_error_analysis.ipynb |
| Sat | Disagreement analysis | Notebook + insights |
| Sun | Buffer / start writeup | Report outline |

### Week 3 (3-9 June): Writeup
| Day | Task | Deliverable |
|-----|------|-------------|
| Mon-Wed | Báo cáo NLP (8-10 trang VN) | Draft v1 |
| Thu | Review & edit | Draft v2 |
| Fri-Sat | Slides 15 phút | Slides draft |
| Sun | Polish | Final docs |

### Week 4 (10-15 June): Submit
| Day | Task |
|-----|------|
| Mon-Wed | Polish, review, fix bugs |
| Thu-Fri | Practice presentation |
| Sat-Sun | Submit |

---

## 5. Risk Management

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| BERT OOM trên máy nhà (4GB) | High | Medium | Train trên máy trường (16GB) hoặc giảm batch size |
| Gemini API rate limit | Medium | Low | Chia nhiều ngày, dùng subset |
| F1-macro thấp hơn paper baseline | Low | High | Reproduce với hyperparameters đúng |
| Domain knowledge thiếu | Medium | Medium | Đọc paper kỹ, hỏi cô Oanh |
| Thời gian không đủ | Medium | High | Buffer 20%, ưu tiên Track A trước |

---

## 6. Deliverables

### Code
- [ ] Modular source code trong `src/`
- [ ] YAML configs cho mọi experiments
- [ ] Unit tests cơ bản
- [ ] CI lint workflow

### Documents
- [ ] README.md (✅)
- [ ] PROJECT_PLAN.md (✅ this file)
- [ ] METHODOLOGY.md
- [ ] EXPERIMENTS.md (log results)
- [ ] REPORT.md (final report)

### Results
- [ ] Model checkpoints (BERT, RoBERTa)
- [ ] Metrics JSON cho mọi experiments
- [ ] Plots (loss curves, confusion matrix, per-class F1)
- [ ] W&B dashboard public link

### Final Submission
- [ ] Báo cáo 8-10 trang tiếng Việt
- [ ] Slides 15 phút
- [ ] Demo notebook
- [ ] Public GitHub repo link

---

## 7. Success Criteria

| Tiêu chí | Mức tối thiểu | Mức tốt | Mức xuất sắc |
|----------|--------------|---------|-------------|
| BERT F1-macro | ≥ 0.40 | ≥ 0.46 (paper) | ≥ 0.50 |
| 4 experiments hoàn thành | 3/4 | 4/4 | 4/4 + ablation |
| Error analysis depth | 20 samples | 50 samples | 50+ + categorization |
| Báo cáo quality | Đủ structure | Có insights | Có publication potential |
| Code quality | Chạy được | Modular, configs | Tests, CI, docs |

---

## 8. Tools & Resources

### Compute
- Máy nhà: GTX 1650 Ti (4GB) — debug, EDA
- Máy trường: RTX 2000 Ada (16GB) — full training (via Parsec)
- Kaggle: 2x T4 (free) — backup nếu cần multi-GPU

### Software
- Python 3.10, PyTorch 2.x
- HuggingFace Transformers, Datasets
- Weights & Biases (tracking)
- Google Gemini API
- Jupyter, VS Code, Git

### Datasets
- GoEmotions (primary)
- (Future) SemEval ABSA — nếu mở rộng sang aspect-based

---

## 9. References

Xem mục References trong [README.md](../README.md).

---

## 10. Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-05-20 | Initial draft | DA |
| 2026-05-21 | Confirm scope: NLP only, BERT vs LLM | DA |
