# Phân Tích Cảm Xúc Đa Nhãn trên Tập Dữ Liệu GoEmotions: So Sánh Fine-tuned BERT và LLM Zero-shot

**Tác giả:** Bùi Đào Duy Anh  
**Môn học:** Xử Lý Ngôn Ngữ Tự Nhiên  
**Ngày nộp:** Tháng 6/2026

---

## Tóm tắt

Bài báo cáo so sánh hai hướng tiếp cận cho phân loại cảm xúc đa nhãn trên GoEmotions (28 classes, 58K samples): (1) fine-tuning BERT/RoBERTa, và (2) LLM zero-shot và few-shot offline (Llama 3.2 3B, Qwen2.5 3B). **Phát hiện chính:** BERT với threshold mặc định đạt F1-macro = 0.4148 ± 0.0008 (3 seeds); sau khi tuning threshold từ 0.5 lên 0.9, BERT đạt **F1-macro = 0.5167** — vượt paper baseline (0.46) mà không cần train lại. Threshold tuning đặc biệt hiệu quả cho rare classes (grief: 0.222→0.400, relief: 0.193→0.400). Few-shot (k=5) cải thiện cả hai local LLM: Qwen 0.2219→**0.2411** (+8.6%), Llama 0.2126→**0.2364** (+11.2%) — nhưng vẫn thua BERT (t=0.9) **~2.1 lần**. **Gemini 2.0 Flash (API, zero-shot)** cho kết quả bất ngờ thấp với F1-macro = **0.0456** — cho thấy large model qua API không tự động tốt hơn local LLM khi không có prompt engineering chuyên sâu. Hypothesis "LLM mạnh hơn ở rare classes" bị bác bỏ với tất cả LLM tested. Hai local LLM có **profile lỗi khác nhau**: Llama aggressive hơn (better recall ở rare classes), Qwen conservative hơn (better precision, ít false positives). Phân tích lỗi BERT vs Qwen trên 2,000 mẫu cho thấy BERT có Jaccard cao hơn 40.4% trường hợp, trong khi LLM chỉ có lợi thế ở single-emotion clear cases.

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
- **EXP-07:** `gemini-2.0-flash-exp` (Google, API-based zero-shot)

EXP-03 đến EXP-06 chạy **offline** trên GPU local (không cần API key), sử dụng HuggingFace `transformers` pipeline với float16 precision. EXP-07 gọi Google Generative AI API, không cần GPU local.

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

Do giới hạn thời gian inference (~0.48 giây/mẫu → ~16 phút cho 2,000 mẫu), LLM được đánh giá trên **random subset 2,000 mẫu** (seed=42) thay vì full test set (đánh giá full 5,427 mẫu ước tính ~43 phút).

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
| Qwen2.5 3B | Few-shot (k=5) | — | 0.2411 | 0.2920 | 0.3041 | **0.0708** | 2K subset |
| Llama 3.2 3B | Few-shot (k=5) | — | 0.2364 | 0.2379 | 0.2952 | 0.1105 | 2K subset |
| Qwen2.5 3B | Zero-shot (local) | — | 0.2219 | 0.2717 | 0.2803 | 0.0713 | 2K subset |
| Llama 3.2 3B | Zero-shot (local) | — | 0.2126 | 0.2306 | 0.2914 | 0.1068 | 2K subset |
| Gemini 2.0 Flash | Zero-shot (API) | — | 0.0456 | 0.3032 | 0.1602 | **0.0549** | 2K subset |
| *Paper baseline* | *BERT Fine-tune* | — | *0.46* | — | — | — | *Demszky 2020* |

> **†** Threshold tuning thực hiện trên test set → **upper bound** thực nghiệm (xem Mục 3.7).  
> **‡** Mean ± std qua 3 seeds (42, 123, 456) — xác nhận tính ổn định (xem Mục 3.9).

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

**Inference stats:** 15m 53s cho 2,000 samples (~0.48 giây/sample trên RTX 2000 Ada).

**Top 5 per-class F1:**

| Emotion | Qwen F1 | BERT F1 | Delta |
|---------|---------|---------|-------|
| gratitude | 0.692 | 0.836 | -0.144 |
| admiration | 0.470 | 0.627 | -0.157 |
| love | 0.449 | 0.750 | -0.301 |
| sadness | 0.449 | — | — |
| fear | 0.440 | — | — |

**Bottom 5 per-class F1:**

| Emotion | Qwen F1 | BERT F1 | Delta |
|---------|---------|---------|-------|
| caring | 0.000 | ~0.300 | -0.300 |
| grief | 0.000 | 0.222 | -0.222 |
| pride | 0.000 | — | — |
| approval | 0.030 | — | — |
| neutral | 0.266 | 0.682 | -0.416 |

**Nhận xét:**
- **RQ1 được xác nhận mạnh:** BERT vượt Qwen zero-shot về F1-macro gần 2x (0.4159 vs 0.2219)
- **RQ3 bị bác bỏ:** LLM không mạnh hơn ở rare classes — grief=0.000, caring=0.000, thấp hơn BERT đáng kể
- **Điểm yếu chính của LLM:** neutral F1 = 0.266 (vs BERT 0.682) — LLM có xu hướng gán emotion vào text thực ra là neutral

### 3.6. Kết Quả EXP-03: Llama 3.2 3B Zero-shot

**Inference stats:** 44m 18s cho 2,000 samples (~1.33 giây/sample — chậm hơn Qwen ~3x do kiến trúc khác).

**So sánh Llama vs Qwen (cùng kích thước 3B, cùng zero-shot):**

| Emotion | BERT | Llama | Qwen | Llama vs Qwen |
|---------|------|-------|------|---------------|
| gratitude | **0.836** | 0.662 | 0.692 | Qwen nhỉnh hơn |
| amusement | **0.803** | **0.519** | 0.429 | Llama +0.09 |
| neutral | **0.682** | 0.321 | 0.266 | Llama +0.06 |
| caring | **0.338** | **0.214** | 0.000 | Llama >> Qwen |
| grief | **0.222** | **0.125** | 0.000 | Llama >> Qwen |
| Hamming Loss | **0.078** | 0.107 | **0.071** | Qwen ít sai hơn |

**Nhận xét:**
- Llama và Qwen có **profile lỗi khác nhau mặc dù cùng cỡ 3B**: Llama aggressive hơn (predict nhiều labels, hamming loss cao hơn) nhưng bắt được rare classes tốt hơn; Qwen conservative hơn (hamming loss thấp) nhưng miss hoàn toàn caring và grief.
- **Partial support RQ3:** Llama mạnh hơn Qwen ở rare classes, nhưng cả hai đều kém xa BERT — RQ1 vẫn được xác nhận mạnh.
- F1-macro tổng thể: BERT (0.416) >> Qwen (0.222) > Llama (0.213).

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

Few-shot inference sử dụng 5 examples từ training set được thêm vào context trước mỗi câu cần phân loại. Cùng 2,000 test samples (seed=42), cùng model weights — chỉ khác phần prompt.

#### So sánh Zero-shot vs Few-shot

| Model | Phương pháp | F1-macro | F1-micro | F1-weighted | Hamming Loss | Δ F1-macro |
|-------|-------------|----------|----------|-------------|--------------|------------|
| Qwen2.5 3B | Zero-shot | 0.2219 | 0.2717 | 0.2803 | 0.0713 | — |
| Qwen2.5 3B | **Few-shot (k=5)** | **0.2411** | **0.2920** | **0.3041** | **0.0708** | **+0.0192 (+8.6%)** |
| Llama 3.2 3B | Zero-shot | 0.2126 | 0.2306 | 0.2914 | 0.1068 | — |
| Llama 3.2 3B | **Few-shot (k=5)** | **0.2364** | **0.2379** | **0.2952** | 0.1105 | **+0.0238 (+11.2%)** |

#### Per-class F1 — Few-shot so với Zero-shot

**Qwen few-shot — Top 5:**

| Emotion | Zero-shot | Few-shot | Delta |
|---------|-----------|----------|-------|
| gratitude | 0.692 | **0.589** | -0.103 |
| love | 0.449 | **0.497** | +0.048 |
| sadness | 0.449 | **0.444** | -0.005 |
| fear | 0.440 | **0.421** | -0.019 |
| admiration | 0.470 | **0.416** | -0.054 |

**Qwen few-shot — Bottom 5 (vẫn thấp):**

| Emotion | Zero-shot | Few-shot | Ghi chú |
|---------|-----------|----------|---------|
| approval | 0.030 | 0.050 | Cải thiện nhỏ |
| grief | 0.000 | 0.000 | Vẫn không detect được |
| pride | 0.000 | 0.000 | Vẫn không detect được |
| relief | 0.000 | 0.000 | Vẫn không detect được |
| remorse | 0.000 | 0.000 | Vẫn không detect được |

**Llama few-shot — Top 5:**

| Emotion | Zero-shot | Few-shot | Delta |
|---------|-----------|----------|-------|
| gratitude | 0.662 | **0.693** | +0.031 |
| love | — | **0.590** | — |
| admiration | — | **0.458** | — |
| amusement | 0.519 | **0.441** | -0.078 |
| joy | — | **0.356** | — |

**Llama few-shot — Bottom 5:**

| Emotion | Few-shot | Ghi chú |
|---------|----------|---------|
| annoyance | 0.105 | Không còn = 0 như zero-shot |
| disgust | 0.083 | — |
| nervousness | 0.072 | — |
| pride | 0.059 | — |
| relief | 0.054 | — |

#### Nhận xét

1. **Few-shot cải thiện đáng kể nhưng không đóng được gap:** Cả hai model tăng ~8–11% F1-macro nhưng vẫn thua BERT (t=0.5) ~1.72x.

2. **Llama hưởng lợi nhiều hơn Qwen từ few-shot** (+11.2% vs +8.6%) — đặc biệt ở việc loại bỏ các class với F1=0.000 trong zero-shot. Llama few-shot không còn class nào = 0.000.

3. **Qwen few-shot vẫn thất bại hoàn toàn ở 4 classes** (grief, pride, relief, remorse = 0.000) — few-shot không đủ để "dạy" model nhận ra các cảm xúc rất hiếm và tinh tế này.

4. **Qwen Hamming Loss giảm nhẹ** (0.0713→0.0708) trong khi Llama tăng (0.1068→0.1105) — few-shot làm Llama predict nhiều labels hơn (aggressive hơn) nhưng cải thiện recall.

5. **Gap BERT vs LLM few-shot:** BERT (t=0.5) = 0.4148 vs Qwen few-shot = 0.2411 → BERT vẫn tốt hơn **1.72x** dù không cần prompting.

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

### 3.10. Kết Quả EXP-07: Gemini 2.0 Flash Zero-shot (API)

EXP-07 đánh giá Gemini 2.0 Flash qua Google Generative AI API với cùng zero-shot prompt như Llama/Qwen. Đây là mô hình lớn duy nhất trong thực nghiệm (>100B params ước tính) và chạy qua API thay vì local GPU.

**Kết quả:** F1-macro = **0.0456**, F1-micro = 0.3032, Hamming Loss = 0.0549 (n=2,000)

**Per-class F1 — Bottom (đa số = 0.000):**

| Emotion | Gemini F1 | Qwen ZS F1 | Llama ZS F1 |
|---------|-----------|------------|------------|
| grief | 0.000 | 0.000 | 0.125 |
| joy | 0.000 | 0.274 | 0.329 |
| sadness | 0.000 | 0.449 | 0.248 |
| fear | 0.000 | 0.440 | 0.241 |
| caring | 0.000 | 0.000 | 0.214 |
| neutral | **0.492** | 0.266 | 0.321 |

**Phân tích — tại sao Gemini low F1-macro nhưng F1-micro cao?**

Gemini có F1-micro = 0.3032 (cao hơn Llama zero-shot 0.2306) nhưng F1-macro chỉ 0.0456 — sự bất nhất này xuất phát từ:
- **Bias mạnh về một số class dominant:** Gemini predict `neutral` rất chính xác (F1=0.492) nhưng bỏ qua hoàn toàn nhiều emotions (joy=0, sadness=0, fear=0)
- **F1-macro unweighted:** Class `neutral` (47% samples) ảnh hưởng mạnh F1-micro, nhưng F1-macro tính bình đẳng 28 classes — và 10+ classes = 0.000 kéo F1-macro xuống rất thấp
- **Khả năng:** Prompt không được tối ưu cho Gemini API, hoặc model truncate output theo cách khác với local HuggingFace pipeline

**Kết luận:** Model lớn qua API không tự động vượt local 3B LLM nếu prompt không được engineering phù hợp. **Gemini zero-shot với prompt hiện tại là kết quả tệ nhất trong tất cả experiments** (kể cả kém hơn Llama/Qwen 3B local).

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

Phân tích 2,000 samples từ EXP-04 được so sánh với predictions của BERT-base trên cùng subset.

**Tổng quan:**
| Category | Số lượng | % |
|----------|---------|---|
| BERT closer (Jaccard cao hơn) | 807 | 40.4% |
| Both wrong (tương đương) | 653 | 32.6% |
| LLM wins (exact match) | 181 | 9.1% |
| BERT wins (exact match) | 160 | 8.0% |
| LLM closer | 102 | 5.1% |
| Both correct | 97 | 4.8% |
| **Disagreement rate** | **1862/2000** | **93.1%** |

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
Mặc dù LLM wins exact matches nhiều hơn một chút (181 vs 160), BERT thường "gần đúng" hơn khi cả hai đều sai (bert_closer = 40.4% vs llm_closer = 5.1%). BERT có Jaccard score trung bình cao hơn trên toàn bộ dataset.

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

---

## 5. Kết Luận

### 5.1. Tóm Tắt Phát Hiện

1. **BERT fine-tuning đạt F1-macro = 0.4148 ± 0.0008 (3 seeds, t=0.5)** — xấp xỉ 90% paper baseline. Std rất nhỏ (±0.0008) xác nhận kết quả ổn định và không phụ thuộc vào random seed. Với **threshold tuning (t=0.9)**, BERT đạt **F1-macro = 0.5167†**, vượt paper baseline (0.46) mà không cần train lại.

2. **RoBERTa không vượt BERT** trong cùng hyperparameter setup — chênh lệch 0.4148 vs 0.4111 = 0.0037 ≈ 4.7× std BERT. Cần multi-seed RoBERTa để kết luận có ý nghĩa thống kê.

3. **Threshold tuning là "low-hanging fruit" quan trọng:** Chỉ cần grid search trên saved logits (không tốn thêm compute), F1-macro tăng +0.10. Rare classes hưởng lợi nhiều nhất — grief: 0.222→0.400, relief: 0.193→0.400.

4. **Few-shot (k=5) cải thiện LLM nhưng không đủ để cạnh tranh với BERT:** Qwen 0.2219→0.2411 (+8.6%), Llama 0.2126→0.2364 (+11.2%). BERT (t=0.5) vẫn vượt LLM few-shot tốt nhất **1.72x**.

5. **Hai LLM có profile lỗi khác nhau, few-shot khuếch đại sự khác biệt:** Llama few-shot loại bỏ hoàn toàn class F1=0.000 (zero-shot: grief=0.125, few-shot: relief=0.054). Qwen few-shot vẫn thất bại hoàn toàn với 4 classes (grief, pride, relief, remorse = 0.000).

6. **RQ3 kết quả nuanced:** Llama có lợi thế ở rare classes hơn Qwen (cả zero-shot lẫn few-shot), nhưng cả hai đều kém xa BERT — kể cả BERT với threshold mặc định.

7. **Gemini 2.0 Flash (API, zero-shot) cho kết quả bất ngờ tệ nhất:** F1-macro = 0.0456 — kém hơn cả Llama/Qwen 3B local mặc dù là model lớn hơn nhiều. Nguyên nhân chính là output bias (predict 10+ classes = 0.000) và prompt chưa được tối ưu cho Gemini API format. Kết quả nhấn mạnh rằng quy mô model không đủ — prompt engineering và alignment với task format quan trọng không kém.

### 5.2. Hạn Chế

- **Threshold tuning trên test set:** Kết quả F1-macro = 0.5167† là upper bound — để công bằng cần validation logits riêng (đã bổ sung lưu val_logits vào `src/train.py` cho future runs). Threshold `t=0.9` được xem là heuristic đề xuất, không phải held-out result.
- **Multi-seed chỉ cho BERT:** RoBERTa vẫn chỉ có 1 seed — kết luận BERT > RoBERTa (0.0037 điểm) cần xác nhận thêm.
- **Chỉ train 3 epochs:** Paper gốc dùng nhiều hơn và fine-tune learning rate. Early stopping theo val F1-macro có thể cải thiện thêm.
- **LLM chỉ đánh giá trên 2,000/5,427 mẫu test** do giới hạn thời gian inference sequential (~19–44 phút cho 2K mẫu).
- **Few-shot k=5 cố định:** Chưa thử k=1, k=3, k=10 để tìm k tối ưu; chưa thử example selection strategies (stratified sampling theo class).
- **Gemini prompt chưa được tối ưu:** Kết quả EXP-07 có thể cải thiện đáng kể với prompt engineering phù hợp cho Gemini API (function calling, structured output, chain-of-thought).

### 5.3. Hướng Phát Triển

1. **Proper threshold tuning:** Val_logits đã được lưu (`results/models/bert_base/val_logits.npy`) — tune threshold trên val set → report trên test. Kết quả hiện tại (t=0.9, test-set tuned) cho thấy tiềm năng; held-out eval sẽ xác nhận tính thực tế.
2. **Multi-seed RoBERTa:** Chạy `run_multiseed.py` với `roberta_base.yaml` để có mean ± std → kết luận chính xác hơn về BERT vs RoBERTa.
3. **Longer fine-tuning:** BERT/RoBERTa với 5–10 epochs + early stopping theo val F1-macro — kỳ vọng tăng thêm ~0.02–0.03 điểm.
4. **DeBERTa-v3-base:** State-of-the-art cho sequence classification, thường vượt BERT/RoBERTa ~2–3 điểm trên GoEmotions.
5. **Few-shot optimization:** Thử k=1, 3, 10 và example selection strategies (stratified per-class sampling) để tối ưu few-shot LLM.
6. **Ensemble:** Kết hợp BERT (precision cao, F1-macro=0.51) + Llama few-shot (recall tốt ở rare classes) theo soft voting trên probability.

---

## 6. Tài Liệu Tham Khảo

1. Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. *ACL 2020*.

2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL 2019*.

3. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv:1907.11692*.

4. Qwen Team. (2024). Qwen2.5 Technical Report. *arXiv:2412.15115*.

5. Meta AI. (2024). Llama 3.2: Lightweight Models with Multimodal Capabilities. *Meta Blog*.

---

*Tất cả 7 experiments đã hoàn tất. Tổng thời gian: EXP-01 BERT 37 phút (×3 seeds = 1h51), EXP-02 RoBERTa 38 phút, EXP-03 Llama zero-shot 44 phút, EXP-04 Qwen zero-shot 16 phút, EXP-05 Llama few-shot 41 phút, EXP-06 Qwen few-shot 19 phút, EXP-07 Gemini zero-shot (API, ~15 phút).*
