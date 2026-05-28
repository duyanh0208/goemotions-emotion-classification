# Methodology

## 1. Dataset

### 1.1 Overview — GoEmotions

GoEmotions (Demszky et al., ACL 2020) is a large-scale, human-annotated emotion dataset derived from English Reddit comments. The **simplified** configuration (used in this project) maps the original taxonomy to **28 classes**: 27 discrete emotions plus *neutral*.

| Split | Samples |
|-------|---------|
| Train | 43,410 |
| Validation | 5,426 |
| Test | 5,427 |
| **Total** | **54,263** |

Each comment was annotated by 3–5 raters; a label is retained when at least 2 raters agree. Texts are short (median ≈ 12 tokens, 95th-percentile ≈ 40 tokens).

### 1.2 Multi-label Nature

Approximately **17 %** of samples carry two or more emotion labels simultaneously (e.g., `["admiration", "gratitude"]`). This rules out standard single-label classifiers; models must produce a **binary vector of length 28** per input.

### 1.3 Class Imbalance

Class frequencies span three orders of magnitude:

| Emotion | Train count (approx.) | Relative frequency |
|---------|----------------------|--------------------|
| neutral | ~13 000 | most common |
| admiration | ~4 500 | frequent |
| grief | ~70 | rare |
| pride | ~80 | rare |
| relief | ~85 | rare |

The imbalance ratio between the most and least frequent class is approximately **184×**. Naive classifiers that always predict the dominant class achieve high *accuracy* but near-zero recall on rare classes — motivating the choice of **F1-macro** as the primary metric, which weights all classes equally regardless of frequency.

---

## 2. Track A — BERT Fine-tuning

### 2.1 Architecture

```
Input text
    │
    ▼
┌──────────────────────────────────────────┐
│  Tokenizer  (WordPiece, max_len = 128)   │
└──────────────────────────────────────────┘
    │  [CLS] t1 t2 … tN [SEP]
    ▼
┌──────────────────────────────────────────┐
│  BERT / RoBERTa encoder                  │
│  (12 layers, 768 hidden, 12 heads)       │
└──────────────────────────────────────────┘
    │  h_[CLS]  ∈ ℝ^768
    ▼
┌──────────────────────────────────────────┐
│  Dropout (p = 0.1)                       │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  Linear: 768 → 28                        │
└──────────────────────────────────────────┘
    │  logits ∈ ℝ^28  (no sigmoid here)
    ▼
BCEWithLogitsLoss (with pos_weight)
```

The `[CLS]` token embedding is used as the sequence representation. A single linear layer maps it to 28 logits — one per emotion class. Sigmoid is applied at inference time to convert logits to probabilities, then thresholded at 0.5 to produce binary predictions.

### 2.2 Training Setup

| Hyperparameter | BERT-base | RoBERTa-base |
|----------------|-----------|--------------|
| Pretrained model | `bert-base-uncased` | `roberta-base` |
| Max sequence length | 128 | 128 |
| Batch size (train) | 16 | 16 |
| Learning rate | 2e-5 | 2e-5 |
| LR scheduler | linear warmup + decay | linear warmup + decay |
| Warmup ratio | 0.1 | 0.1 |
| Epochs | 3 | 3 |
| Dropout | 0.1 | 0.1 |
| Optimizer | AdamW | AdamW |
| Weight decay | 0.01 | 0.01 |
| Gradient clipping | 1.0 | 1.0 |
| Random seed | 42 | 42 |

### 2.3 Loss Function — BCEWithLogitsLoss with `pos_weight`

Standard binary cross-entropy weights all classes equally, which leads to a model that ignores rare emotions. To counteract class imbalance, each class `c` is assigned a **positive weight**:

```
pos_weight[c] = (N − n_c) / n_c
```

where `N` is the total number of training samples and `n_c` is the number of samples labelled with class `c`. This weight scales the loss contribution of the positive (present) class upward relative to the negative (absent) class.

Weights are clipped to `[1.0, 50.0]` to prevent numerical instability for the rarest classes (grief, pride, relief) where unconstrained weights would exceed 600.

The weighted BCE loss is:

```
L = − Σ_c  w_c · [y_c · log σ(z_c) + (1 − y_c) · log(1 − σ(z_c))]
```

where `w_c = pos_weight[c]`, `y_c ∈ {0,1}` is the ground-truth label, and `z_c` is the raw logit.

---

## 3. Track B — LLM In-context Learning

### 3.1 Model

**Gemini 2.0 Flash** (`gemini-2.0-flash`) is used via the Google Generative AI SDK. It is accessed through the free-tier API (15 requests/minute limit). Temperature is fixed at **0.0** for deterministic outputs; `response_mime_type = "application/json"` enforces structured JSON output.

### 3.2 Zero-shot vs Few-shot

| Setting | Context given | Expected advantage |
|---------|---------------|--------------------|
| Zero-shot | Task instruction + emotion list | No labelled data needed |
| Few-shot (k=5) | Zero-shot prompt + 5 examples | Anchors output format; may improve rare-class recall |

The 5 few-shot examples are **fixed and manually curated** — they are not selected dynamically from the training set. They were chosen to cover:
- Positive, multi-label (e.g., excitement + pride)
- Negative, multi-label (e.g., anger + disgust)
- Neutral (no emotion)
- Rare class (grief + sadness)
- Mild surprise + curiosity

### 3.3 Prompt Design Rationale

Key decisions:

1. **Explicit multi-label instruction**: The prompt says *"select ALL emotions that apply"* to prevent the model from defaulting to single-label output.

2. **Closed vocabulary**: The prompt lists all 28 class names and instructs the model to use only those strings. This eliminates hallucinated labels (e.g., "melancholy" instead of "sadness").

3. **Neutral fallback**: The instruction explicitly says to use `["neutral"]` when no emotion is discernible — matching the dataset convention and preventing empty predictions.

4. **JSON-only output**: Gemini is configured with `response_mime_type="application/json"` plus an explicit instruction to output only JSON. This avoids parsing ambiguity from free-text explanations.

5. **Temperature = 0**: Deterministic sampling ensures reproducible results for evaluation. Variation from sampling noise would confound comparison.

---

## 4. Evaluation Protocol

### 4.1 Metrics

| Metric | Formula | Why used |
|--------|---------|----------|
| **F1-macro** | Mean of per-class F1 | Primary metric — weights all classes equally; penalises failure on rare classes |
| **F1-micro** | TP_all / (TP_all + 0.5·(FP_all + FN_all)) | Dominated by frequent classes; useful secondary view |
| **F1-weighted** | Class-frequency-weighted mean F1 | Even more dominated by frequent classes |
| **Hamming Loss** | Fraction of wrong label assignments | Captures overall binary accuracy across all class slots |

**Why F1-macro is primary:** With 184× class imbalance, F1-micro and accuracy are misleading — a model can score F1-micro = 0.60 while completely ignoring rare classes. F1-macro forces the model to perform well on grief, pride, and relief, which are scientifically interesting and practically important.

### 4.2 Threshold Selection

For BERT/RoBERTa, sigmoid outputs are thresholded at **0.5** (default). This is the standard baseline. An alternative is to tune the threshold per-class on the validation set; this is noted as a potential improvement but not the primary evaluation setting.

For LLM inference, there is no threshold — the model directly outputs emotion names.

### 4.3 Evaluation Samples

Due to API rate limits (15 RPM, free tier), LLM evaluation is run on a **random subset of 2,000 test samples** (seed=42). Fine-tuned models are evaluated on the **full test set of 5,427 samples**. Results on the 2,000-sample subset are reported separately for fair comparison.

---

## 5. Experimental Setup

### 5.1 Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA (see `nvidia-smi` output) |
| CUDA | 12.x |
| RAM | 16 GB+ |
| OS | Windows 11 / Linux |

### 5.2 Reproducibility

All experiments use **random seed = 42** throughout:
- Python `random.seed(42)`
- `numpy.random.seed(42)`
- `torch.manual_seed(42)` + `torch.cuda.manual_seed_all(42)`
- `torch.backends.cudnn.deterministic = True`
- Dataset subsampling uses `numpy.random.seed(42)`

LLM inference uses `temperature=0.0` for deterministic output.

### 5.3 Software Versions

| Library | Version |
|---------|---------|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| Transformers | 4.30+ |
| datasets | 2.x |
| scikit-learn | 1.x |
| google-generativeai | 0.8+ |
| numpy | 1.24+ |

Full pinned versions are in `requirements.txt`.

### 5.4 Baseline Reference

The paper (Demszky et al., 2020) reports **BERT-base F1-macro = 0.46** on the simplified 28-class GoEmotions test set. This serves as the baseline for comparison. Reproducing ≥ 0.46 with the fine-tuning track is a sanity-check milestone.
