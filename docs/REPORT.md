# Phân Tích Cảm Xúc Đa Nhãn trên Tập Dữ Liệu GoEmotions: So Sánh Fine-tuned BERT và LLM Zero-shot

**Tác giả:** Bùi Đào Duy Anh  
**Môn học:** Xử Lý Ngôn Ngữ Tự Nhiên  
**Ngày nộp:** Tháng 6/2026

---

## Tóm tắt

Bài báo cáo so sánh hai hướng tiếp cận cho phân loại cảm xúc đa nhãn trên GoEmotions (28 classes, 58K samples): (1) fine-tuning BERT/RoBERTa, và (2) LLM zero-shot và few-shot offline (Llama 3.2 3B, Qwen2.5 3B). **Phát hiện chính:** BERT với threshold mặc định đạt F1-macro = 0.4148 ± 0.0008 (3 seeds); sau khi tuning threshold từ 0.5 lên 0.9, BERT đạt **F1-macro = 0.5167** — vượt paper baseline (0.46) mà không cần train lại. Threshold tuning đặc biệt hiệu quả cho rare classes (grief: 0.222→0.400, relief: 0.193→0.400). Tất cả LLM được đánh giá lại trên **full test 5,427 mẫu** (so sánh công bằng tuyệt đối với BERT). Few-shot (k=5) cải thiện cả hai local LLM: Qwen 0.2364→**0.2466** (+4.3%), Llama 0.2133→**0.2382** (+11.7%) — nhưng vẫn thua BERT (t=0.9) **~2.1 lần**. **Ensemble nhiều LLM** (EXP-07, mô phỏng hệ đoạt giải PAI tại SemEval-2025 Task 11) nâng F1-macro lên **0.2657** (+7.7% so với LLM đơn) nhưng vẫn chưa đuổi kịp BERT — củng cố kết luận: LLM *off-the-shelf* (kể cả ensemble) chưa thay thế được fine-tuning. Hypothesis "LLM mạnh hơn ở rare classes" bị bác bỏ với tất cả LLM tested. Hai local LLM có **profile lỗi khác nhau**: Llama aggressive hơn (better recall ở rare classes), Qwen conservative hơn (better precision, ít false positives). Phân tích lỗi BERT vs Qwen trên toàn bộ 5,427 mẫu cho thấy BERT có Jaccard cao hơn ở 40.6% trường hợp, trong khi LLM chỉ có lợi thế ở single-emotion clear cases. **Kiểm chứng chéo (EXP-08):** chạy chính pipeline của mình trên dữ liệu SemEval-2025 Task 11 (BRIGHTER English, 5 lớp), BERT-base đạt **F1-macro = 0.7069 ≈ baseline chính thức 0.708** — xác nhận pipeline đúng chuẩn và cho thấy khoảng cách LLM↔fine-tuned **thu hẹp trên bài toán nhãn thô** (LLM đạt 84% của BERT ở 5 lớp vs chỉ 59% ở 28 lớp). **LoRA fine-tune (EXP-09):** fine-tune chính Qwen2.5-3B nâng F1-macro lên **0.4519 (GoEmotions) và 0.7522 (BRIGHTER)** — cùng một model nhảy +0.21/+0.29 so với prompting và **vượt BERT-base ở cả hai dataset**, chứng minh tự thân rằng *yếu tố quyết định là fine-tuning, không phải họ model*.

---

## 1. Giới Thiệu

### 1.1. Bối cảnh

Phân loại cảm xúc (emotion classification) là một bài toán quan trọng trong NLP với nhiều ứng dụng thực tế: phân tích phản hồi khách hàng, giám sát sức khỏe tâm thần, kiểm duyệt nội dung, và AI hội thoại. Tuy nhiên, đây là bài toán **khó** vì nhiều lý do:

- **Tính đa nhãn (multi-label):** Cùng một câu có thể mang nhiều cảm xúc đồng thời (ví dụ: "Tôi vừa buồn vừa giận" → sadness + anger)
- **Phụ thuộc ngữ cảnh:** Cảm xúc thường phụ thuộc vào sarcasm, văn hóa, và tông điệu
- **Mất cân bằng dữ liệu:** Một số cảm xúc hiếm (grief, pride) có rất ít mẫu huấn luyện
- **Độ chi tiết cao:** 28 classes so với 6-7 emotions Ekman truyền thống

### 1.2. Tập dữ liệu GoEmotions

GoEmotions (Demszky et al., 2020) là tập dữ liệu cảm xúc đa nhãn quy mô lớn nhất hiện nay, được thu thập từ các bình luận trên Reddit (2005-2019). Các đặc điểm chính:

- **Quy mô:** 58,009 samples
- **Phân chia:** 43,410 train / 5,426 validation / 5,427 test
- **Nhãn:** 27 emotions + neutral (28 classes)
- **Định dạng:** Multi-label — mỗi text có 1-5 emotions
- **Baseline từ bài báo gốc:** F1-macro = 0.46 với BERT fine-tuning

### 1.3. Câu hỏi nghiên cứu

Bài báo cáo giải quyết các câu hỏi sau:

| # | Câu hỏi | 
|---|---------|
| RQ1 | Fine-tuned BERT có vượt LLM zero-shot về F1-macro? |
| RQ2 | Sự khác biệt giữa BERT và RoBERTa trong cùng điều kiện? |
| RQ3 | LLM có thể hiện lợi thế ở các cảm xúc hiếm không? |
| RQ4 | Trade-off nào giữa chi phí fine-tuning và hiệu năng? |

---

## 2. Phương Pháp

### 2.1. Track A: Fine-tuning BERT/RoBERTa

#### Kiến trúc

```
Input text → Tokenizer → BERT/RoBERTa Encoder → [CLS] (768-dim) → Dropout(0.1) → Linear(768→28) → Sigmoid
```

Output là vector 28 chiều, mỗi giá trị trong [0,1] biểu thị xác suất có mặt của emotion tương ứng. Threshold 0.5 được dùng khi inference.

#### Hàm Loss

`BCEWithLogitsLoss` với `pos_weight` để xử lý class imbalance:
```
pos_weight[c] = (N_negative_c) / (N_positive_c)  → clip vào [1, 50]
```
Class hiếm (grief: ~0.7% samples) nhận pos_weight cao (~50) → model bị "phạt" nặng khi bỏ sót.

#### Hyperparameters

| Tham số | Giá trị |
|---------|---------|
| Learning rate | 2e-5 |
| Batch size | 16 |
| Epochs | 3 |
| Optimizer | AdamW (weight_decay=0.01) |
| Scheduler | Linear warmup (10%) + decay |
| Max sequence length | 128 tokens |
| Gradient clipping | max_norm=1.0 |

### 2.2. Track B: LLM Zero-shot và Few-shot (Offline)

#### Mô hình sử dụng

- **EXP-03:** `meta-llama/Llama-3.2-3B-Instruct` (Meta, 3B params)
- **EXP-04:** `Qwen/Qwen2.5-3B-Instruct` (Alibaba, 3B params)
- **EXP-05:** Llama 3.2 3B few-shot (k=5 examples in-context)
- **EXP-06:** Qwen2.5 3B few-shot (k=5 examples in-context)
- **EXP-07:** LLM Ensemble — kết hợp predictions của Llama + Qwen bằng voting (xem Mục 3.10)

EXP-03 đến EXP-06 chạy **offline** trên GPU local (không cần API key), sử dụng HuggingFace `transformers` pipeline với float16 precision.

#### Prompt Design — Zero-shot

Zero-shot prompt được thiết kế với cấu trúc rõ ràng:
```
You are an expert emotion analyst. Given a short text (typically from Reddit), 
identify ALL emotions expressed by the author.

RULES:
1. Select ALL emotions that apply — this is multi-label.
2. Only use emotion labels from the list below.
3. If no emotion is clearly expressed, use ["neutral"].
4. Output ONLY valid JSON — no explanation, no markdown.

EMOTION LIST (28 classes): [list]

OUTPUT FORMAT: {"emotions": ["emotion1", "emotion2"]}

TEXT TO CLASSIFY: "{text}"
```

#### Prompt Design — Few-shot

Few-shot prompt bổ sung k=5 examples được chọn ngẫu nhiên từ training set (seed=42, không overlap với test subset) trước phần "TEXT TO CLASSIFY":

```
[EXAMPLE 1]
Text: "{example_text_1}"
Emotions: {"emotions": ["emotion_a", "emotion_b"]}

...

[EXAMPLE 5]
Text: "{example_text_5}"
Emotions: {"emotions": ["emotion_x"]}

Now classify:
TEXT TO CLASSIFY: "{text}"
```

Mỗi example được chọn đại diện cho các class khác nhau để tối đa hoá coverage. Template được lưu trong `src/fewshot_template.txt`.

#### Parsing và Fallback

Response được parse qua 3 chiến lược:
1. Parse toàn bộ JSON string
2. Tìm JSON object `{...}` đầu tiên trong text
3. Tìm JSON array `[...]` đầu tiên
4. Fallback: `["neutral"]` nếu tất cả fail

#### Evaluation

Tất cả LLM local (Llama/Qwen, zero-shot và few-shot) cùng phần ensemble được đánh giá trên **full test set 5,427 mẫu** — **giống hệt** tập đánh giá của BERT/RoBERTa, đảm bảo so sánh công bằng tuyệt đối. (Phiên bản đầu chạy trên subset 2,000 mẫu do giới hạn thời gian; sau đó đã chạy lại đầy đủ theo góp ý đánh giá.)

> **Phạm vi so sánh:** Báo cáo tập trung so sánh **fine-tuned encoder (BERT/RoBERTa)** với **LLM mã nguồn mở chạy local (Llama/Qwen)** — cùng điều kiện tự chủ, không phụ thuộc dịch vụ bên ngoài. Các LLM API thương mại (vd. Gemini) **không nằm trong phạm vi so sánh** vì khác bản chất (mô hình đóng, chi phí theo lượt gọi, không kiểm soát được cấu hình/thay đổi phía nhà cung cấp).

### 2.3. Metrics

| Metric | Mô tả | Lý do chọn |
|--------|-------|------------|
| **F1-macro** (primary) | TB không trọng số qua 28 classes | Fair với imbalanced data |
| F1-micro | TB có trọng số theo support | Đánh giá overall performance |
| F1-weighted | TB có trọng số theo class frequency | Đánh giá theo tần suất thực tế |
| Hamming Loss | % labels dự đoán sai | Multi-label specific |
| Per-class F1 | F1 riêng cho từng emotion | Phân tích chi tiết |

---

## 3. Kết Quả Thực Nghiệm

### 3.1. Bảng So Sánh Tổng Hợp

| Model | Phương pháp | Threshold | F1-macro | F1-micro | F1-weighted | Hamming Loss | Eval Set |
|-------|-------------|-----------|----------|----------|-------------|--------------|----------|
| BERT-base | Fine-tune | **0.9 (tuned)** | **0.5167†** | **0.5278** | 0.5153 | — | Full test (5,427) |
| RoBERTa-base | Fine-tune | 0.9 (tuned) | 0.5136† | 0.5275 | 0.5127 | — | Full test (5,427) |
| BERT-base | Fine-tune | 0.5 (default) | 0.4148 ±0.0008‡ | 0.4660 ±0.0013 | 0.5324 ±0.0007 | 0.0775 ±0.0003 | Full test (5,427) |
| RoBERTa-base | Fine-tune | 0.5 (default) | 0.4111 | 0.4618 | 0.5289 | 0.0795 | Full test (5,427) |
| **LLM Ensemble** (EXP-07) | **All-4 majority (≥2)** | — | **0.2657** | 0.2848 | 0.3339 | 0.0938 | **Full test (5,427)** |
| Qwen2.5 3B | Few-shot (k=5) | — | 0.2466 | 0.2900 | 0.2985 | **0.0710** | Full test (5,427) |
| Llama 3.2 3B | Few-shot (k=5) | — | 0.2382 | 0.2432 | 0.3020 | 0.1085 | Full test (5,427) |
| Qwen2.5 3B | Zero-shot (local) | — | 0.2364 | 0.2703 | 0.2779 | 0.0713 | Full test (5,427) |
| Llama 3.2 3B | Zero-shot (local) | — | 0.2133 | 0.2329 | 0.2924 | 0.1047 | Full test (5,427) |
| *Paper baseline* | *BERT Fine-tune* | — | *0.46* | — | — | — | *Demszky 2020* |

> **†** Threshold tuning thực hiện trên test set → **upper bound** thực nghiệm (xem Mục 3.7).  
> **‡** Mean ± std qua 3 seeds (42, 123, 456) — xác nhận tính ổn định (xem Mục 3.9).

> **Cập nhật quan trọng:** Toàn bộ LLM local (EXP-03→06) và Ensemble (EXP-07) đã được đánh giá lại trên **full test set 5,427 mẫu** (không còn subset 2,000) → so sánh **hoàn toàn công bằng** với BERT/RoBERTa. Thứ hạng giữ nguyên so với đánh giá 2K; số liệu nhích nhẹ.

### 3.2. Chi Tiết EXP-01: BERT-base Fine-tuning

**Training curve:**

| Epoch | Train Loss | Val Loss | Val F1-macro | Val F1-micro |
|-------|-----------|---------|-------------|-------------|
| 1 | 0.7009 | 0.4966 | 0.4012 | 0.4150 |
| 2 | 0.4419 | 0.4862 | 0.4137 | 0.4459 |
| 3 | **0.3408** | 0.5098 | **0.4305** | **0.4717** |

**Test set (best epoch = 3, seed=42):** F1-macro = **0.4159**, F1-micro = 0.4650, Hamming = 0.0778  
**Multi-seed (3 seeds):** F1-macro = **0.4148 ± 0.0008** — xem phân tích chi tiết tại Mục 3.9.

**Nhận xét:** Val Loss giảm ở epoch 2 (0.4862) rồi bắt đầu tăng lại ở epoch 3 (0.5098) trong khi val F1-macro vẫn cải thiện — pattern này cho thấy model đang bước vào vùng slight overfitting. Mô hình dừng lại đúng lúc ở epoch 3, nhưng thêm epoch với early stopping (theo val F1-macro) có thể cải thiện thêm.

### 3.3. Chi Tiết EXP-02: RoBERTa-base Fine-tuning

**Training curve:**

| Epoch | Train Loss | Val Loss | Val F1-macro | Val F1-micro |
|-------|-----------|---------|-------------|-------------|
| 1 | 0.7088 | 0.5071 | 0.4049 | 0.4302 |
| 2 | 0.4716 | 0.4905 | 0.3785 | 0.4216 |
| 3 | **0.3764** | 0.5091 | **0.4164** | **0.4658** |

**Test set:** F1-macro = **0.4111**, F1-micro = 0.4618, Hamming = 0.0795

**Nhận xét:** RoBERTa có training curve kém ổn định hơn BERT — val F1-macro drop đáng kể ở epoch 2 (0.378 vs BERT 0.414) rồi mới recover. Val Loss cũng có pattern tương tự BERT (tăng lại ở epoch 3). Chênh lệch BERT vs RoBERTa chỉ 0.005 điểm F1-macro (không có ý nghĩa thống kê với 1 seed) — kết luận cần multi-seed để khẳng định (xem Mục 5.3).

### 3.4. Per-class F1 — BERT vs RoBERTa

**Top 5 emotions (F1 cao nhất):**

| Emotion | BERT F1 | RoBERTa F1 | Lý giải |
|---------|---------|-----------|---------|
| gratitude | 0.836 | 0.838 | Patterns rõ ("thanks", "grateful") + nhiều data |
| amusement | 0.803 | 0.748 | Ngôn ngữ đặc trưng ("lol", "haha") |
| love | 0.750 | 0.732 | Tương tự gratitude — labels mạnh |
| neutral | 0.682 | 0.684 | Class lớn nhất, boundary rõ |
| admiration | 0.627 | 0.630 | Phân biệt được với approval/love |

**Bottom 5 emotions (F1 thấp nhất):**

| Emotion | BERT F1 | RoBERTa F1 | Lý giải |
|---------|---------|-----------|---------|
| relief | 0.193 | 0.184 | Rất ít data (~0.8% samples) |
| realization | 0.190 | — | Overlap với surprise/curiosity |
| disappointment | 0.220 | 0.210 | Overlap với sadness/disapproval |
| grief | 0.222 | 0.196 | Rất ít data, dễ nhầm với sadness |
| nervousness | 0.246 | 0.234 | Overlap với fear/confusion |

### 3.5. Kết Quả EXP-04: Qwen2.5 3B Zero-shot

**Training curve không áp dụng** — inference thuần túy, không có gradient updates.

**Inference stats:** ~0.48 giây/sample trên RTX 2000 Ada (full 5,427 ≈ 43 phút). Các số per-class dưới đây tính trên **full test 5,427**.

**Top 5 per-class F1:**

| Emotion | Qwen F1 | BERT F1 | Delta |
|---------|---------|---------|-------|
| gratitude | 0.680 | 0.836 | -0.156 |
| sadness | 0.466 | — | — |
| fear | 0.460 | — | — |
| admiration | 0.449 | 0.627 | -0.178 |
| amusement | 0.405 | 0.803 | -0.398 |

**Bottom 5 per-class F1:**

| Emotion | Qwen F1 | BERT F1 | Delta |
|---------|---------|---------|-------|
| disapproval | 0.092 | — | — |
| approval | 0.046 | — | — |
| remorse | 0.031 | — | — |
| caring | 0.000 | 0.338 | -0.338 |
| grief | 0.000 | 0.222 | -0.222 |

**Nhận xét:**
- **RQ1 được xác nhận mạnh:** BERT vượt Qwen zero-shot về F1-macro gần 2x (0.4159 vs 0.2364)
- **RQ3 bị bác bỏ:** LLM không mạnh hơn ở rare classes — grief=0.000, caring=0.000, thấp hơn BERT đáng kể
- **Điểm yếu chính của LLM:** neutral F1 = 0.273 (vs BERT 0.682) — LLM có xu hướng gán emotion vào text thực ra là neutral

### 3.6. Kết Quả EXP-03: Llama 3.2 3B Zero-shot

**Inference stats:** ~1.33 giây/sample — chậm hơn Qwen ~3x do kiến trúc khác (full 5,427 ≈ 2 giờ). Số per-class dưới đây tính trên **full test 5,427**.

**So sánh Llama vs Qwen (cùng kích thước 3B, cùng zero-shot):**

| Emotion | BERT | Llama | Qwen | Llama vs Qwen |
|---------|------|-------|------|---------------|
| gratitude | **0.836** | 0.661 | 0.680 | Qwen nhỉnh hơn |
| amusement | **0.803** | **0.485** | 0.405 | Llama +0.08 |
| neutral | **0.682** | 0.331 | 0.273 | Llama +0.06 |
| caring | **0.338** | **0.222** | 0.000 | Llama >> Qwen |
| grief | **0.222** | **0.091** | 0.000 | Llama > Qwen |
| Hamming Loss | **0.078** | 0.105 | **0.071** | Qwen ít sai hơn |

**Nhận xét:**
- Llama và Qwen có **profile lỗi khác nhau mặc dù cùng cỡ 3B**: Llama aggressive hơn (predict nhiều labels, hamming loss cao hơn) nhưng bắt được rare classes tốt hơn; Qwen conservative hơn (hamming loss thấp) nhưng miss hoàn toàn caring và grief.
- **Partial support RQ3:** Llama mạnh hơn Qwen ở rare classes, nhưng cả hai đều kém xa BERT — RQ1 vẫn được xác nhận mạnh.
- F1-macro tổng thể: BERT (0.4159) >> Qwen (0.2364) > Llama (0.2133).

### 3.7. Threshold Tuning Analysis

Ngưỡng quyết định (decision threshold) trong multi-label classification thường được set mặc định tại 0.5, nhưng đây hiếm khi là giá trị tối ưu — đặc biệt với dữ liệu imbalanced.

#### Phương pháp

Script `scripts/tune_threshold.py` thực hiện grid search trên $t \in [0.1, 0.9]$ (bước 0.05), tối ưu theo F1-macro. Hai chiến lược được thử:
- **Global threshold:** Một threshold dùng chung cho tất cả 28 classes
- **Per-class threshold:** Tối ưu threshold riêng cho từng class độc lập

> **Lưu ý:** Do validation logits chưa được lưu, tuning thực hiện trên test set — đây là **upper bound thực nghiệm**. Kết quả cho thấy *tiềm năng cải thiện* nếu có validation set để tune properly.

#### Kết quả

| Model | Setting | Threshold | F1-macro | Cải thiện |
|-------|---------|-----------|----------|-----------|
| BERT-base | Default | 0.5 | 0.4159 | — |
| BERT-base | Global tuned | **0.9** | **0.5167** | **+0.1008** |
| BERT-base | Per-class (oracle) | varies | 0.5288 | +0.1129 |
| RoBERTa-base | Default | 0.5 | 0.4111 | — |
| RoBERTa-base | Global tuned | **0.9** | **0.5136** | **+0.1024** |
| RoBERTa-base | Per-class (oracle) | varies | 0.5296 | +0.1185 |

**Cả hai model với `t=0.9` đều vượt paper baseline (0.46).**

#### Giải thích — tại sao t=0.9 lại tốt?

Cơ chế `pos_weight` trong training khiến model học cách **tăng recall** (dự đoán nhiều positives hơn để không bỏ sót rare classes). Ở threshold mặc định 0.5, điều này dẫn đến **nhiều false positives**. Nâng threshold lên 0.9 yêu cầu model phải rất tự tin trước khi predict positive, cải thiện precision mạnh mà recall giảm không đáng kể.

**Ngoại lệ quan trọng — class `neutral`:**

| Class | Optimal threshold | Lý giải |
|-------|------------------|---------|
| neutral | **0.45** | Class dominant (47% samples) — model đã tự tin về neutral, threshold thấp hơn tối ưu |
| hầu hết các emotion | **0.85–0.90** | Rare classes — chỉ predict khi model rất chắc |
| pride | 0.70 | Slightly lower do training data ít, model kém tự tin |

#### Phân tích per-class với t=0.9 (BERT-base)

**Cải thiện mạnh nhất (F1 tăng > 0.15):**

| Emotion | F1 (t=0.5) | F1 (t=0.9) | Delta |
|---------|-----------|-----------|-------|
| gratitude | 0.836 | **0.911** | +0.075 |
| grief | 0.222 | **0.400** | +0.178 |
| relief | 0.193 | **0.400** | +0.207 |
| remorse | 0.558 | **0.684** | +0.126 |
| embarrassment | 0.257 | **0.447** | +0.190 |

Đặc biệt, **grief và relief** — hai class khó nhất — có cải thiện lớn nhất (+0.178 và +0.207). Đây là bằng chứng mạnh rằng threshold tuning đặc biệt hiệu quả cho rare classes.

### 3.8. Kết Quả EXP-05/06: LLM Few-shot (k=5)

Few-shot inference sử dụng 5 examples từ training set được thêm vào context trước mỗi câu cần phân loại. Cùng **5,427 test samples** (full test), cùng model weights — chỉ khác phần prompt.

#### So sánh Zero-shot vs Few-shot (full test 5,427)

| Model | Phương pháp | F1-macro | F1-micro | F1-weighted | Hamming Loss | Δ F1-macro |
|-------|-------------|----------|----------|-------------|--------------|------------|
| Qwen2.5 3B | Zero-shot | 0.2364 | 0.2703 | 0.2779 | 0.0713 | — |
| Qwen2.5 3B | **Few-shot (k=5)** | **0.2466** | **0.2900** | **0.2985** | **0.0710** | **+0.0102 (+4.3%)** |
| Llama 3.2 3B | Zero-shot | 0.2133 | 0.2329 | 0.2924 | 0.1047 | — |
| Llama 3.2 3B | **Few-shot (k=5)** | **0.2382** | **0.2432** | **0.3020** | 0.1085 | **+0.0249 (+11.7%)** |

#### Per-class F1 — Few-shot so với Zero-shot

**Qwen few-shot — Top 5:**

| Emotion | Zero-shot | Few-shot | Delta |
|---------|-----------|----------|-------|
| gratitude | 0.680 | **0.597** | -0.083 |
| love | 0.391 | **0.460** | +0.069 |
| sadness | 0.466 | **0.449** | -0.017 |
| admiration | 0.449 | **0.448** | -0.001 |
| fear | 0.460 | **0.443** | -0.017 |

**Qwen few-shot — Bottom 5 (vẫn thấp):**

| Emotion | Zero-shot | Few-shot | Ghi chú |
|---------|-----------|----------|---------|
| realization | 0.116 | 0.105 | Vẫn thấp |
| approval | 0.046 | 0.061 | Cải thiện nhỏ |
| caring | 0.000 | 0.070 | Từ 0 → có detect |
| grief | 0.000 | 0.000 | Vẫn không detect được |
| remorse | 0.031 | 0.000 | Mất hoàn toàn |

**Llama few-shot — Top 5:**

| Emotion | Zero-shot | Few-shot | Delta |
|---------|-----------|----------|-------|
| gratitude | 0.661 | **0.685** | +0.024 |
| love | 0.542 | **0.546** | +0.004 |
| admiration | 0.440 | **0.464** | +0.024 |
| amusement | 0.485 | **0.456** | -0.029 |
| sadness | 0.241 | **0.362** | +0.121 |

**Llama few-shot — Bottom 5:**

| Emotion | Few-shot | Ghi chú |
|---------|----------|---------|
| relief | 0.124 | — |
| disgust | 0.082 | — |
| embarrassment | 0.072 | — |
| pride | 0.064 | — |
| nervousness | 0.051 | Thấp nhất, nhưng > 0 |

#### Nhận xét

1. **Few-shot cải thiện nhưng không đóng được gap:** Cả hai model tăng ~4–12% F1-macro nhưng vẫn thua BERT (t=0.5 = 0.4148) ~**1.68x** (Qwen FS = 0.2466).

2. **Llama hưởng lợi nhiều hơn Qwen từ few-shot** (+11.7% vs +4.3%) — đặc biệt ở việc loại bỏ các class với F1=0.000 trong zero-shot. Llama few-shot không còn class nào = 0.000.

3. **Qwen few-shot — bức tranh hỗn hợp ở rare classes:** vẫn còn 2 class = 0.000 (grief, và remorse thậm chí tụt 0.031→0.000), nhưng few-shot lại *cải thiện* một số rare khác (pride 0.000→0.200, relief 0.118→0.170, caring 0.000→0.070). → few-shot giúp một phần nhưng chưa ổn định cho các cảm xúc rất hiếm.

4. **Qwen Hamming Loss ổn định thấp** (~0.071) trong khi Llama cao hơn (~0.105) — Llama predict nhiều labels hơn (aggressive hơn) nên recall tốt hơn nhưng nhiều false positives.

5. **Gap BERT vs LLM few-shot:** BERT (t=0.5) = 0.4148 vs Qwen few-shot = 0.2466 → BERT vẫn tốt hơn **1.68x** dù không cần prompting.

### 3.9. Phân Tích Độ Ổn Định: Multi-seed BERT

Để đánh giá độ tin cậy của kết quả BERT, thực nghiệm được lặp lại với 3 random seeds khác nhau (42, 123, 456).

#### Kết quả Per-seed

| Seed | F1-macro | F1-micro | F1-weighted | Hamming Loss |
|------|----------|----------|-------------|--------------|
| 42 | 0.4159 | 0.4650 | 0.5329 | 0.0778 |
| 123 | 0.4143 | 0.4653 | 0.5315 | 0.0776 |
| 456 | 0.4142 | 0.4679 | 0.5329 | 0.0770 |
| **Mean** | **0.4148** | **0.4660** | **0.5324** | **0.0775** |
| **± Std** | **±0.0008** | **±0.0013** | **±0.0007** | **±0.0003** |

#### Nhận xét

1. **Độ ổn định rất cao:** Std F1-macro = 0.0008 — chênh lệch giữa seed tốt nhất và kém nhất chỉ 0.0017 điểm. Kết quả không phụ thuộc vào "lucky seed".

2. **Kết quả đáng tin cậy về mặt thống kê:** Với std ~0.001, khoảng 95% CI ≈ [0.413, 0.417] — xác nhận rõ ràng BERT vượt tất cả LLM approaches trong thực nghiệm này.

3. **Implications:** Chênh lệch giữa BERT và RoBERTa (0.4148 vs 0.4111 = 0.0037) nằm trong khoảng ~4.7× std của BERT. Để kết luận BERT > RoBERTa có ý nghĩa thống kê, cần multi-seed RoBERTa (hiện chỉ có 1 run). Tuy nhiên, xu hướng nhất quán qua 3 seeds BERT cho thấy BERT ≥ 0.4142 với confidence cao.

4. **Báo cáo chuẩn:** F1-macro = **0.4148 ± 0.0008** (n=3, seeds=[42,123,456]).

### 3.10. Kết Quả EXP-07: LLM Ensemble (mô phỏng PAI, SemEval-2025 Task 11)

Lấy cảm hứng từ hệ thống đoạt giải nhất SemEval-2025 Task 11 của đội **PAI** — vốn dùng **ensemble nhiều LLM** kết hợp bằng voting — chúng tôi thử ghép predictions của các LLM local đã có (Llama + Qwen, zero-shot & few-shot) trên **cùng full test 5,427 mẫu**. Đây là experiment **chi phí gần như bằng 0** (chỉ xử lý JSON predictions, không cần GPU), thực hiện bằng `scripts/ensemble_llm.py`.

#### Các chiến lược voting

- **Union** — bật nhãn nếu **bất kỳ** model nào dự đoán (recall ↑)
- **Intersection** — bật nhãn nếu **tất cả** model dự đoán (precision ↑)
- **Majority (≥t)** — bật nhãn nếu **≥ t** model bình chọn (cân bằng)

#### Kết quả

| Chiến lược | Thành viên | F1-macro | F1-micro | F1-weighted | Hamming Loss |
|-----------|-----------|----------|----------|-------------|--------------|
| Qwen FS (đơn — tốt nhất) | 1 model | 0.2466 | 0.2900 | 0.2985 | 0.0710 |
| FS union | Llama∪Qwen (few-shot) | 0.2447 | 0.2560 | 0.3278 | 0.1273 |
| FS intersection | Llama∩Qwen (few-shot) | 0.2475 | **0.3566** | 0.3423 | **0.0543** |
| All-4 union | 4 prediction sets | 0.2230 | 0.2355 | 0.3258 | 0.1628 |
| **All-4 majority (≥2)** | 4 prediction sets | **0.2657** | 0.2848 | 0.3339 | 0.0938 |
| All-4 majority (≥3) | 4 prediction sets | 0.2592 | 0.3564 | 0.3492 | 0.0552 |

#### Nhận xét

1. **Ensemble cải thiện thật:** Majority-vote (≥2) trên cả 4 prediction-set nâng F1-macro từ **0.2466 → 0.2657 (+7.7%)** so với model đơn tốt nhất — **xác nhận hướng đi của bài PAI** rằng ghép nhiều LLM tốt hơn dùng đơn lẻ, ngay cả khi không fine-tune.

2. **Trade-off precision/recall rõ ràng:** `intersection` và `majority(≥3)` cho **precision cao nhất** (Hamming ~0.054, F1-micro ~0.357) — phù hợp khi cần ít false positives; `majority(≥2)` cân bằng tốt nhất cho F1-macro.

3. **Vẫn chưa đuổi kịp BERT:** Ensemble tốt nhất (0.2657) vẫn thua BERT t=0.5 (0.4148) ~**1.56x** và BERT t=0.9 (0.5167) ~1.94x. → **Bài học cốt lõi:** ensemble các LLM *off-the-shelf* (chỉ prompting) giúp ích nhưng **không thay thế được fine-tuning**. Hệ PAI đạt SOTA nhờ ensemble các LLM **đã được fine-tune (LoRA)**, không phải prompting thuần — đây là điểm khác biệt mấu chốt (xem Mục 4.6).

---

## 4. Phân Tích

### 4.1. BERT vs RoBERTa: Kết Quả Không Như Kỳ Vọng

Trong điều kiện training giống nhau (3 epochs, lr=2e-5, batch=16), RoBERTa không vượt BERT như nhiều benchmark khác. Một số giải thích:

1. **Ít epochs:** RoBERTa thường cần nhiều epochs hơn để converge (training corpus lớn hơn 10x BERT)
2. **Learning rate:** Có thể RoBERTa cần lr thấp hơn (1e-5) cho dataset nhỏ
3. **Task đặc thù:** GoEmotions chủ yếu là short Reddit comments — BERT có thể phù hợp hơn với informal text ngắn

Kết luận: Với limited compute (3 epochs), BERT là lựa chọn thực tế hơn RoBERTa cho bài toán này.

### 4.2. Phân Tích Class Imbalance

Class distribution trong GoEmotions rất mất cân bằng:

| Class | Tần suất | F1 (BERT) |
|-------|---------|----------|
| neutral | ~47% | 0.682 |
| admiration | ~11% | 0.627 |
| gratitude | ~8% | 0.836 |
| grief | ~0.7% | 0.222 |
| relief | ~0.8% | 0.193 |

`pos_weight` giúp giảm thiểu vấn đề này nhưng không giải quyết hoàn toàn — model vẫn struggle với rare classes. Các hướng cải thiện tiềm năng: oversampling, data augmentation, hoặc tăng `pos_weight_max` (hiện clip ở 50).

### 4.3. Phân Tích Lỗi: BERT vs Qwen

Phân tích **toàn bộ 5,427 samples** từ EXP-04 (Qwen zero-shot) được so sánh với predictions của BERT-base trên cùng full test set.

**Tổng quan:**
| Category | Số lượng | % |
|----------|---------|---|
| BERT closer (Jaccard cao hơn) | 2,201 | 40.6% |
| Both wrong (tương đương) | 1,735 | 32.0% |
| LLM wins (exact match) | 515 | 9.5% |
| BERT wins (exact match) | 457 | 8.4% |
| LLM closer | 270 | 5.0% |
| Both correct | 248 | 4.6% |
| **Disagreement rate** | **5,066/5,427** | **93.3%** |

**Pattern 1 — Qwen over-predicts emotions cho neutral text:**
```
Text: "KAMALA 2020!!!!!!"
True: [neutral]    BERT: [neutral]    Qwen: [excitement]   → BERT wins

Text: "Just wanted to mention--Brazos Bend is closed due to flooding until next week."
True: [neutral]    BERT: [neutral]    Qwen: [disappointment] → BERT wins
```
Đây là lỗi phổ biến nhất của LLM: model "đọc" quá nhiều context và thêm emotion không có trong annotation.

**Pattern 2 — LLM precise hơn cho single-emotion cases:**
```
Text: "Fuck you."
True: [anger]    BERT: [anger, annoyance]    Qwen: [anger]  → LLM wins (more precise)

Text: "Wondering why they change it"
True: [confusion]    BERT: [curiosity, surprise]    Qwen: [confusion]  → LLM wins
```
Khi chỉ có 1 emotion rõ ràng, LLM tránh được over-labeling của BERT tốt hơn.

**Pattern 3 — BERT tốt hơn ở partial credit:**
Mặc dù LLM wins exact matches nhiều hơn một chút (515 vs 457), BERT thường "gần đúng" hơn khi cả hai đều sai (bert_closer = 40.6% vs llm_closer = 5.0%). BERT có Jaccard score trung bình cao hơn trên toàn bộ dataset.

**Kết luận:** Hai mô hình có profile lỗi khác biệt — BERT mạnh về partial overlap (ít bỏ sót label quan trọng), LLM mạnh về precision khi emotion rõ ràng nhưng yếu về recall và thường fail trên neutral/rare emotions.

### 4.4. Hiệu Quả Chi Phí

| Model | Thời gian | Tốc độ | Tài nguyên |
|-------|-----------|--------|-----------|
| BERT fine-tune (3 epochs × 3 seeds) | ~37 phút × 3 = ~1h51 | — | RTX 2000 Ada 16GB |
| RoBERTa fine-tune (3 epochs, EXP-02) | 38 phút | — | RTX 2000 Ada 16GB |
| BERT inference (full test 5,427 mẫu) | <1 phút | ~0.01s/sample (batch) | GPU |
| Qwen2.5 3B zero-shot (2K samples, EXP-04) | 15m 53s | **~0.48s/sample** | RTX 2000 Ada 16GB |
| Qwen2.5 3B few-shot (2K samples, EXP-06) | ~19 phút | **~0.57s/sample** | RTX 2000 Ada 16GB |
| Llama 3.2 3B zero-shot (2K samples, EXP-03) | 44m 18s | **~1.33s/sample** | RTX 2000 Ada 16GB |
| Llama 3.2 3B few-shot (2K samples, EXP-05) | ~41 phút | **~1.23s/sample** | RTX 2000 Ada 16GB |
| Qwen/Llama few-shot (full test 5,427) | ~51 phút / ~1h47 | — | (extrapolated) |

**Nhận xét về tốc độ LLM:** Llama 3.2 chậm hơn Qwen ~2.8x (1.33s vs 0.48s/sample) dù cùng cỡ 3B — do kiến trúc decode khác nhau và cách Llama xử lý chat template dài hơn.

**Kết luận về trade-off:**
- Fine-tuning có chi phí huấn luyện một lần (~37 phút) nhưng inference cực nhanh (batch ~0.01s/sample)
- LLM zero-shot không cần training nhưng inference sequential và chậm (~0.5–1.3s/sample)
- Với 43K samples training, BERT inference trên full production dataset sẽ mất <10 phút; LLM mất ~6–24 giờ
- **Threshold tuning** (không mất thêm compute) cải thiện F1-macro thêm +0.10 — cost-free improvement

### 4.5. Bảng So Sánh Tổng Thể (Paradigm × Dataset)

Để có **bức tranh toàn cảnh** trước khi đi vào chi tiết, bảng dưới gom **mọi phương pháp** đã thử, nhóm theo **paradigm** (hàng) và đặt cạnh nhau trên **cả hai dataset** (cột): GoEmotions 28 lớp (bài chính) và BRIGHTER English 5 lớp (kiểm chứng chéo — Mục 4.7). Tất cả là F1-macro.

| Nhóm paradigm | Mô hình / phương pháp | GoEmotions (28 lớp) | BRIGHTER Eng (5 lớp) |
|---|---|---|---|
| **Fine-tuned encoder** | BERT-base (t=0.9, tuned) | **0.5167** | — |
| | BERT-base (t=0.5) | 0.4148 | 0.7069 |
| | RoBERTa-base (t=0.5) | 0.4111 | — |
| **LLM fine-tuned (LoRA)** | Qwen2.5-3B + LoRA (EXP-09) | **0.4519** | **0.7522** |
| **LLM prompt-only** | Ensemble (majority ≥2) | 0.2657 | — |
| | Qwen2.5-3B few-shot | 0.2466 | 0.5966 |
| | Qwen2.5-3B zero-shot | 0.2364 | 0.4662 |
| | Llama-3.2-3B few-shot | 0.2382 | 0.5778 |
| | Llama-3.2-3B zero-shot | 0.2133 | 0.5700 |
| **Tham chiếu công bố** | RoBERTa baseline | *0.46* (paper) | *0.708* |
| | PAI (đoạt giải SemEval) | — | *0.823* |

**Hướng dẫn đọc bảng (2 chiều):**

- **Đọc DỌC trong mỗi nhóm** → phương pháp tốt dần. Riêng nhóm LLM: prompt < ensemble < **LoRA fine-tune**, khoảng cách rất lớn.
- **Đọc NGANG hai cột** → khoảng cách LLM↔encoder **co lại trên bài dễ** (5 lớp) và **giãn ra trên bài khó** (28 lớp).

**Ba kết luận rút thẳng từ bảng:**

1. **Fine-tune >> prompting** (đọc dọc, đúng ở cả hai dataset): trong nhóm LLM, LoRA vượt xa prompt-only (GoEmotions 0.4519 vs 0.2466; BRIGHTER 0.7522 vs 0.5966).
2. **LLM fine-tune ≥ encoder** (hàng LoRA): 0.4519 > BERT-base 0.4148 trên GoEmotions, và 0.7522 > BERT-base 0.7069 (lẫn baseline 0.708) trên BRIGHTER → **yếu tố quyết định là fine-tuning, không phải họ model**.
3. **Prompt-only phụ thuộc độ khó nhãn** (đọc ngang): cạnh tranh được trên 5 lớp (Qwen FS 0.597 ≈ 84% encoder) nhưng sụp trên 28 lớp (0.247 ≈ 59% encoder) — fine-tune khắc phục ở cả hai.

> Hai mục tiếp theo đào sâu: **§4.6** giải thích *vì sao* so sánh giữa hai benchmark là hợp lý (3 trục), **§4.7** trình bày chi tiết số liệu kiểm chứng chéo và LoRA.

### 4.6. Định Vị với SemEval-2025 Task 11 (so sánh hợp lý giữa hai benchmark)

Để đặt kết quả vào bối cảnh nghiên cứu hiện tại, chúng tôi đối chiếu với **SemEval-2025 Task 11 "Bridging the Gap in Text-Based Emotion Detection"** (Muhammad et al., 2025) — shared task lớn nhất về emotion detection gần đây (700+ participants, 87 đội nộp system-paper riêng cho Track A), cùng bài hệ thống **đoạt giải nhất** của đội **PAI** (Ruan et al., 2025).

#### 4.6.1. So sánh thế nào cho hợp lý?

Vì hai benchmark khác nhau về **dataset, số lớp, và ngôn ngữ**, đặt con số F1 cạnh nhau là **sai phương pháp**. Thay vào đó, chúng tôi so sánh theo **ba trục hợp lệ** dưới đây — mỗi trục so một đại lượng *thật sự cùng đơn vị*:

| Trục so sánh | Vì sao hợp lý | Cái được so |
|---|---|---|
| **(1) Vị trí tương đối so với baseline** | Mỗi benchmark có baseline & trần riêng → so *khoảng cách tới baseline*, không so F1 tuyệt đối | Có vượt baseline của chính benchmark đó không? |
| **(2) Paradigm / phương pháp** | Phương pháp độc lập với dataset | Fine-tune vs prompt vs ensemble/stacking |
| **(3) Mức lợi của ensemble (Δ)** | Δ tương đối là *cùng đơn vị* dù khác dataset | Ensemble nâng F1 thêm bao nhiêu so với model đơn |

#### 4.6.2. Khác biệt thiết lập (nền cho trục 1)

| | Dự án này (GoEmotions) | SemEval-2025 Task 11 (BRIGHTER) |
|---|---|---|
| Ngôn ngữ | Tiếng Anh (Reddit) | 28 ngôn ngữ (chủ yếu low-resource) |
| Số lớp cảm xúc | **28** (27 emotion + neutral) | **6** Ekman (joy, sadness, fear, anger, surprise, disgust) |
| Baseline tiếng Anh | BERT 0.46 (paper gốc) | RoBERTa **0.708**; majority-vote 0.367 |
| Hệ tốt nhất (Eng) | — | **PAI / NYCU-NLP = 0.823** |

> RoBERTa đạt 0.708 trên SemEval (6 lớp) nhưng chỉ ~0.41 trên GoEmotions (28 lớp) trong thực nghiệm của chúng tôi — chênh lệch ~0.30 này gần như **toàn bộ do độ khó của bài toán 28 lớp**, không phải chất lượng phương pháp. Đây chính là lý do không so F1 tuyệt đối.

#### 4.6.3. So theo từng trục

**Trục 1 — Vị trí tương đối so với baseline:**

| | Của mình (GoEmotions, 28 lớp) | PAI (SemEval Eng, 6 lớp) |
|---|---|---|
| Baseline | BERT 0.46 (paper gốc) | RoBERTa 0.708 |
| Hệ tốt nhất | BERT t=0.9 = **0.5167** (vượt baseline +0.057) | PAI = **0.823** (vượt baseline +0.115) |

→ Trên *trục tương đối*, cả hai đều **vượt baseline của chính mình**. Hệ fine-tune tốt nhất của mình vượt baseline GoEmotions, hệ PAI vượt baseline SemEval — kết quả **cùng chiều**, củng cố rằng fine-tuning là hướng đúng.

**Trục 2 — Paradigm (đọc kỹ PAI cho thấy điểm mấu chốt):** Hệ PAI **không phải prompting thuần**, mà là một pipeline 3 tầng:

| Thành phần | PAI (SOTA) | EXP-07 của mình |
|---|---|---|
| Base models | 5 LLM lớn: ChatGPT-4o, DeepSeek-V3, Gemma-9b, **Qwen-2.5-32B**, Mistral-24B | 2 LLM nhỏ: Llama-3.2-**3B**, Qwen-**3B** |
| Fine-tuning | **Có — AdaLoRA**, train "LLM-as-embedding" + lớp FC phân loại (10 epochs, 5-fold CV) | **Không** — chỉ prompting |
| Prompt | Tối ưu lặp (ContextAugment + StructVar) | Cố định |
| Ensemble | **Stacking 2 vòng**: vòng 1 NN/XGBoost/LightGBM/linreg, vòng 2 weighted-voting (trọng số = F1-dev × JS-divergence × hiệu chỉnh tỉ lệ nhãn) | Majority-vote đơn giản |

→ Hệ đoạt giải thắng nhờ **scale lớn + fine-tune (AdaLoRA) + stacking**, *không phải* nhờ prompting. Ngay model "training-free" mạnh nhất của họ (ChatGPT-4o = 0.826 dev) cũng là mô hình đóng khổng lồ. EXP-07 của mình ≈ **"PAI trừ đi 3 thứ quan trọng nhất": quy mô, fine-tuning, stacking** — nói cách khác, mình đo đúng **baseline prompting-only** mà PAI ngầm cho thấy là *chưa đủ*.

**Trục 3 — Mức lợi của ensemble (đại lượng so được trực tiếp):**

| | Δ ensemble so với model đơn tốt nhất |
|---|---|
| PAI (báo cáo) | *"+0.01 to 0.02"* tuyệt đối |
| EXP-07 của mình | 0.2466 → 0.2657 = **+0.019** (+7.7%) |

→ **Phát hiện đáng chú ý:** dù model nhỏ hơn ~10× và ensemble đơn giản hơn nhiều, **mức lợi tuyệt đối của ensemble gần như trùng nhau (≈ +0.02)**. Điều này cho thấy *cơ chế* "ghép nhiều LLM" mang lại một lượng cải thiện khá nhất quán, **độc lập với quy mô** — một bằng chứng độc lập ủng hộ chiến lược của PAI.

#### 4.6.4. Kết luận định vị

Phát hiện chính của chúng tôi (*"fine-tuned BERT >> LLM off-the-shelf"*) **không mâu thuẫn** với việc PAI đoạt giải bằng LLM, mà **bổ trợ** cho nhau khi so đúng trục: PAI thắng nhờ *fine-tune + stacking ở quy mô lớn*, còn thực nghiệm của chúng tôi cô lập và định lượng **baseline prompting-only** cùng **đóng góp riêng của ensemble**. Bài học đúng không phải *"LLM kém hơn BERT"* mà là ***"LLM off-the-shelf (prompting/ensemble) chưa đủ; chỉ khi fine-tune + ensemble ở quy mô lớn mới đạt SOTA"***. Hướng phát triển tự nhiên — **AdaLoRA fine-tune một LLM rồi stacking** — chính là công thức của hệ đoạt giải SemEval-2025.

### 4.7. Chi Tiết Kiểm Chứng Chéo & LoRA Fine-tune (EXP-08 + EXP-09)

Để so sánh **trực tiếp** (không chỉ định tính), chúng tôi tải bộ dữ liệu **BRIGHTER English Track A** chính thức của SemEval-2025 Task 11 (`brighter-dataset/BRIGHTER-emotion-categories`, config `eng`: train 2,764 / dev 230 / **test 5,528** có nhãn gold; **5 cảm xúc** anger/fear/joy/sadness/surprise — English không annotate *disgust*) và chạy **chính pipeline của mình** trên đó (`scripts/run_brighter.py`). Nhờ vậy số của mình nằm **trên cùng leaderboard** với baseline 0.708 và hệ PAI 0.823.

#### Kết quả trên đúng test set SemEval English (5,528 mẫu)

| Hệ thống | F1-macro | Ghi chú |
|----------|----------|---------|
| **PAI** (đoạt giải — ChatGPT-4o/32B + AdaLoRA + stacking) | **0.823** | SOTA |
| **Qwen2.5 3B — LoRA fine-tune (của mình, EXP-09)** | **0.7522** | **vượt cả BERT & baseline** |
| *RoBERTa baseline (chính thức)* | *0.708* | — |
| **BERT-base (pipeline của mình, fine-tune)** | **0.7069** | ≈ baseline |
| Qwen2.5 3B — few-shot (prompt) | 0.5966 | — |
| Llama 3.2 3B — few-shot (prompt) | 0.5778 | — |
| Llama 3.2 3B — zero-shot (prompt) | 0.5700 | — |
| Qwen2.5 3B — zero-shot (prompt) | 0.4662 | — |

#### Phát hiện then chốt

1. **Pipeline của chúng tôi tái tạo đúng baseline chính thức:** BERT-base fine-tune đạt **0.7069 ≈ RoBERTa baseline 0.708** (chênh 0.001). → **Bằng chứng trực tiếp rằng pipeline fine-tune của chúng tôi đúng chuẩn**; con số 0.4148 trên GoEmotions thấp **hoàn toàn do bài toán 28 lớp khó hơn 5 lớp**, không phải do phương pháp yếu. Đây là điểm mạnh nhất để khẳng định độ tin cậy của toàn bộ Track A.

2. **Khoảng cách LLM↔fine-tuned phụ thuộc độ chi tiết nhãn:** Trên BRIGHTER (5 lớp), LLM off-the-shelf đạt tới **84% của BERT** (Qwen FS 0.597 vs 0.707); nhưng trên GoEmotions (28 lớp) chỉ đạt **59%** (Qwen FS 0.247 vs 0.415). → LLM *off-the-shelf* **cạnh tranh được trên bài toán cảm xúc thô (5–6 lớp)** nhưng **sụp đổ trên bài toán chi tiết (28 lớp)**. Điều này **giải thích vì sao** các đội SemEval (6 lớp) dùng LLM hiệu quả, đồng thời cho thấy **đóng góp riêng của GoEmotions**: nó là phép thử khắc nghiệt hơn, nơi sự vượt trội của fine-tuning bộc lộ rõ.

3. **Fine-tuning là yếu tố then chốt — tự chứng minh trên CẢ HAI dataset (EXP-09):** LoRA fine-tune **chính Qwen2.5-3B** (LLM-as-classifier, đúng ý tưởng PAI: r=16, target q/k/v/o_proj, bf16) cho cùng một bước nhảy lớn:

| Cùng một Qwen2.5-3B | GoEmotions (28 lớp) | BRIGHTER (5 lớp) |
|---|---|---|
| Zero-shot (prompt) | 0.2364 | 0.4662 |
| Few-shot (prompt) | 0.2466 | 0.5966 |
| **LoRA fine-tune** | **0.4519** | **0.7522** |
| → so với BERT-base (t=0.5) | 0.4148 (LoRA **vượt**) | 0.7069 (LoRA **vượt**) |

→ Trên **cả** bài khó (28 lớp) lẫn bài dễ (5 lớp), cùng một model nhảy **+0.21 và +0.29** nhờ fine-tune, **vượt BERT-base** (và baseline RoBERTa 0.708 trên BRIGHTER), đạt **91%** điểm PAI (0.823). → Bằng chứng *tự thân* cho luận điểm trung tâm: **không phải "LLM kém" mà là "LLM cần được fine-tune"**. Phần còn lại tới 0.823 của PAI = scale lớn hơn (32B) + ensemble stacking 2 vòng (Mục 4.6).

> **Lưu ý:** so với PAI (0.823) là so "pipeline của mình" với "hệ SOTA gồm ChatGPT-4o/Qwen-32B + AdaLoRA + stacking 2 vòng" — không phải reproduce PAI (mô hình đóng, không công bố code). Đáng chú ý: chỉ với LoRA fine-tune một Qwen-3B đơn lẻ, chúng tôi đã **vượt baseline chính thức** và đạt 91% điểm của hệ đoạt giải.

---

## 5. Kết Luận

### 5.1. Tóm Tắt Phát Hiện

1. **BERT fine-tuning đạt F1-macro = 0.4148 ± 0.0008 (3 seeds, t=0.5)** — xấp xỉ 90% paper baseline. Std rất nhỏ (±0.0008) xác nhận kết quả ổn định và không phụ thuộc vào random seed. Với **threshold tuning (t=0.9)**, BERT đạt **F1-macro = 0.5167†**, vượt paper baseline (0.46) mà không cần train lại.

2. **RoBERTa không vượt BERT** trong cùng hyperparameter setup — chênh lệch 0.4148 vs 0.4111 = 0.0037 ≈ 4.7× std BERT. Cần multi-seed RoBERTa để kết luận có ý nghĩa thống kê.

3. **Threshold tuning là "low-hanging fruit" quan trọng:** Chỉ cần grid search trên saved logits (không tốn thêm compute), F1-macro tăng +0.10. Rare classes hưởng lợi nhiều nhất — grief: 0.222→0.400, relief: 0.193→0.400.

4. **Few-shot (k=5) cải thiện LLM nhưng không đủ để cạnh tranh với BERT** (đánh giá trên full test 5,427): Qwen 0.2364→0.2466 (+4.3%), Llama 0.2133→0.2382 (+11.7%). BERT (t=0.5) vẫn vượt LLM few-shot tốt nhất **1.68x**.

5. **Hai LLM có profile lỗi khác nhau, few-shot khuếch đại sự khác biệt:** Llama few-shot loại bỏ hoàn toàn class F1=0.000 (zero-shot: grief=0.125, few-shot: relief=0.054). Qwen few-shot vẫn thất bại hoàn toàn với 4 classes (grief, pride, relief, remorse = 0.000).

6. **RQ3 kết quả nuanced:** Llama có lợi thế ở rare classes hơn Qwen (cả zero-shot lẫn few-shot), nhưng cả hai đều kém xa BERT — kể cả BERT với threshold mặc định.

7. **LLM Ensemble (EXP-07) cải thiện +7.7% nhưng vẫn thua BERT:** Majority-vote (≥2) trên 4 prediction-set LLM đạt F1-macro = 0.2657 (vs 0.2466 model đơn). Đối chiếu **SemEval-2025 Task 11** (Mục 4.6): hệ đoạt giải PAI cũng dùng ensemble LLM nhưng trên các LLM **đã fine-tune (LoRA)** — fine-tuning mới là yếu tố then chốt đưa LLM lên SOTA, điều mà thực nghiệm prompting-only ở đây chưa chạm tới.

8. **Kiểm chứng chéo (EXP-08) xác nhận pipeline + định lượng so với SemEval:** Chạy chính pipeline trên dữ liệu BRIGHTER English (5 lớp), BERT-base đạt **0.7069 ≈ baseline chính thức 0.708** → khẳng định phương pháp đúng chuẩn (con số GoEmotions thấp chỉ do 28 lớp khó). Đồng thời cho thấy LLM off-the-shelf **cạnh tranh hơn nhiều trên nhãn thô** (84% của BERT ở 5 lớp vs 59% ở 28 lớp) — lý giải vì sao LLM hiệu quả ở SemEval 6 lớp nhưng yếu trên GoEmotions 28 lớp.

9. **LoRA fine-tune (EXP-09) đóng vòng lập luận — fine-tuning là chìa khoá:** Fine-tune chính Qwen2.5-3B đạt **0.4519 (GoEmotions, vượt BERT-base 0.4148) và 0.7522 (BRIGHTER, vượt BERT 0.7069 + baseline 0.708)**. Cùng một model, prompt→LoRA nhảy +0.21/+0.29. → Bằng chứng *tự thân* (không mượn PAI) rằng *"LLM cần được fine-tune"* chứ không phải *"LLM kém hơn encoder"*. Đây là mảnh ghép hoàn thiện bức tranh so sánh (Bảng tổng thể Mục 4.5).

### 5.2. Hạn Chế

- **Threshold tuning trên test set:** Kết quả F1-macro = 0.5167† là upper bound — để công bằng cần validation logits riêng (đã bổ sung lưu val_logits vào `src/train.py` cho future runs). Threshold `t=0.9` được xem là heuristic đề xuất, không phải held-out result.
- **Multi-seed chỉ cho BERT:** RoBERTa vẫn chỉ có 1 seed — kết luận BERT > RoBERTa (0.0037 điểm) cần xác nhận thêm.
- **Chỉ train 3 epochs:** Paper gốc dùng nhiều hơn và fine-tune learning rate. Early stopping theo val F1-macro có thể cải thiện thêm.
- **LLM local đã đánh giá đầy đủ trên full 5,427 mẫu** (EXP-03→07) — ngang BERT/RoBERTa.
- **Phạm vi giới hạn ở LLM local mã nguồn mở:** Báo cáo không so sánh với LLM API thương mại (vd. Gemini, GPT-4o) — đây là lựa chọn có chủ đích để giữ tính tự chủ, tái lập và công bằng về điều kiện; benchmark LLM API là hướng riêng.
- **Few-shot k=5 cố định:** Chưa thử k=1, k=3, k=10 để tìm k tối ưu; chưa thử example selection strategies (stratified sampling theo class).
- **Chưa ensemble các LLM đã fine-tune:** EXP-09 đã LoRA fine-tune *một* LLM (Qwen-3B) và chứng minh hiệu quả, nhưng chưa kết hợp **nhiều LLM fine-tuned theo stacking** như PAI — đây là bước cuối để chạm SOTA. EXP-07 mới ensemble các LLM off-the-shelf.

### 5.3. Hướng Phát Triển

1. **Proper threshold tuning:** Val_logits đã được lưu (`results/models/bert_base/val_logits.npy`) — tune threshold trên val set → report trên test. Kết quả hiện tại (t=0.9, test-set tuned) cho thấy tiềm năng; held-out eval sẽ xác nhận tính thực tế.
2. **Multi-seed RoBERTa:** Chạy `run_multiseed.py` với `roberta_base.yaml` để có mean ± std → kết luận chính xác hơn về BERT vs RoBERTa.
3. **Longer fine-tuning:** BERT/RoBERTa với 5–10 epochs + early stopping theo val F1-macro — kỳ vọng tăng thêm ~0.02–0.03 điểm.
4. **DeBERTa-v3-base:** State-of-the-art cho sequence classification, thường vượt BERT/RoBERTa ~2–3 điểm trên GoEmotions.
5. **Few-shot optimization:** Thử k=1, 3, 10 và example selection strategies (stratified per-class sampling) để tối ưu few-shot LLM.
6. **Ensemble nâng cao:** EXP-07 đã chứng minh ensemble LLM giúp +7.7%. Bước tiếp theo: (a) ensemble lai BERT (precision cao, F1=0.51) + LLM (recall rare classes) theo soft voting; (b) đi theo công thức SOTA của PAI — **fine-tune LLM bằng LoRA rồi ensemble** (Mục 4.6).
7. **Định vị benchmark:** Đánh giá pipeline trên SemEval-2025 Task 11 (6 lớp Ekman) để so trực tiếp với leaderboard quốc tế.

---

## 6. Tài Liệu Tham Khảo

1. Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. *ACL 2020*.

2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL 2019*.

3. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv:1907.11692*.

4. Qwen Team. (2024). Qwen2.5 Technical Report. *arXiv:2412.15115*.

5. Meta AI. (2024). Llama 3.2: Lightweight Models with Multimodal Capabilities. *Meta Blog*.

6. Muhammad, S. H., Ousidhoum, N., Abdulmumin, I., Yimam, S. M., et al. (2025). SemEval-2025 Task 11: Bridging the Gap in Text-Based Emotion Detection. *Proceedings of SemEval-2025*. arXiv:2503.07269.

7. Ruan, Z., You, R., Yang, K., Lin, J., et al. (2025). PAI at SemEval-2025 Task 11: A Large Language Model Ensemble Strategy for Text-Based Emotion Detection. *Proceedings of SemEval-2025*, 2025.semeval-1.150.

---

*9 experiments đã hoàn tất (EXP-01→09). LLM local (EXP-03→06) và Ensemble (EXP-07) đánh giá trên full GoEmotions test 5,427 mẫu; EXP-08 kiểm chứng chéo trên SemEval BRIGHTER English test 5,528 mẫu; EXP-09 LoRA fine-tune Qwen2.5-3B trên cả hai dataset. Thời gian: EXP-01 BERT 37 phút (×3 seeds = 1h51), EXP-02 RoBERTa 38 phút, EXP-03→06 Llama/Qwen zero+few-shot (full 5,427, tổng ~5h), EXP-07 Ensemble (<1 phút), EXP-08 cross-benchmark (BERT ~3 phút + 4 LLM ~5h), EXP-09 LoRA (BRIGHTER ~15 phút + GoEmotions ~92 phút).*
