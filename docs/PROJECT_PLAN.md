# 📋 Project Plan — GoEmotions Emotion Classification

> **Document version:** 2.0 (June 2026)
> **Author:** Bùi Đào Duy Anh
> **Status:** ✅ Hoàn tất — 9 thí nghiệm (EXP-01 → EXP-09), 2 dataset (GoEmotions + SemEval-2025 BRIGHTER)

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
| RQ1 | Fine-tuned BERT có vượt LLM zero/few-shot? | So sánh F1-macro trên cùng test 5,427 |
| RQ2 | BERT vs RoBERTa trong cùng điều kiện? | So sánh fine-tune đồng cấu hình |
| RQ3 | LLM mạnh ở rare classes? | Per-class F1 analysis |
| RQ4 | Trade-off accuracy vs cost? | Train-time / inference-time analysis |

> **Mở rộng (theo góp ý GVHD):** đối chiếu với **SemEval-2025 Task 11** — chạy chính pipeline trên dữ liệu BRIGHTER (EXP-08) và LoRA fine-tune một LLM (EXP-09) để đóng vòng lập luận.

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

### 3.3. Track B: LLM In-context Learning (mã nguồn mở, chạy local)

**Models:** (chạy local, không cần API key — bảo đảm tái lập & công bằng điều kiện)
- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`

> LLM API thương mại (vd. Gemini) **ngoài phạm vi** so sánh vì khác bản chất (mô hình đóng, chi phí theo lượt gọi, không kiểm soát cấu hình).

**Zero-shot prompt:** đóng vai chuyên gia cảm xúc, ràng buộc multi-label, chỉ dùng 28 nhãn, fallback `["neutral"]`, xuất CHỈ JSON `{"emotions":[...]}`. Giải mã greedy (`do_sample=False`).

**Few-shot prompt (k=5):** thêm 5 ví dụ từ training set (seed=42) trước câu cần phân loại.

**Evaluation:** **full test 5,427 mẫu** — giống hệt Track A (so sánh công bằng tuyệt đối).

### 3.3b. Track C: LLM Fine-tune & Ensemble

- **EXP-07 Ensemble:** majority-vote 4 prediction-set Llama/Qwen (`scripts/ensemble_llm.py`).
- **EXP-08 Cross-benchmark:** chạy pipeline trên BRIGHTER English (SemEval-2025 Task 11) — `scripts/run_brighter.py`.
- **EXP-09 LoRA fine-tune:** LoRA fine-tune Qwen2.5-3B làm classifier (`scripts/lora_finetune.py`, peft r=16, bf16).

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
- [x] Model checkpoints (BERT, RoBERTa, LoRA-Qwen) — *local, không commit (quá lớn)*
- [x] Metrics JSON cho mọi experiments (`results/metrics/`)
- [x] Plots (per-class F1, model comparison)

### Final Submission
- [x] Báo cáo kết quả đầy đủ ([REPORT.md](REPORT.md)) + log thí nghiệm ([EXPERIMENTS.md](EXPERIMENTS.md))
- [x] Public GitHub repo link

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
- HuggingFace Transformers, Datasets, PEFT (LoRA)
- scikit-learn (metrics)
- Jupyter, VS Code, Git

### Datasets
- GoEmotions (primary, 28 lớp)
- SemEval-2025 Task 11 — BRIGHTER English (kiểm chứng chéo, 5 lớp)

---

## 9. References

Xem mục References trong [README.md](../README.md).

---

## 10. Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-05-20 | Initial draft | DA |
| 2026-05-21 | Confirm scope: NLP only, BERT vs LLM | DA |
| 2026-06-26 | v2.0 — chốt 9 thí nghiệm, LLM local (Llama/Qwen) thay Gemini, full 5,427; thêm SemEval-2025 cross-benchmark (EXP-08) + LoRA (EXP-09) | DA |
