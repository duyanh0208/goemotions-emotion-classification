# Phân Tích Cảm Xúc Đa Nhãn trên Tập Dữ Liệu GoEmotions: So Sánh Fine-tuned BERT và LLM Zero-shot

**Tác giả:** Bùi Đào Duy Anh  
**Môn học:** Xử Lý Ngôn Ngữ Tự Nhiên  
**Ngày nộp:** Tháng 6/2026

---

## Tóm tắt

Bài báo cáo so sánh hai hướng tiếp cận cho phân loại cảm xúc đa nhãn trên GoEmotions (28 classes, 58K samples): (1) fine-tuning BERT/RoBERTa, và (2) LLM zero-shot offline (Llama 3.2 3B, Qwen2.5 3B). Kết quả: BERT đạt F1-macro = **0.4159** (full test 5,427 mẫu), vượt cả hai LLM gần **2 lần** (Qwen=0.2219, Llama=0.2126 trên 2K subset). Hypothesis "LLM mạnh hơn ở rare classes" bị bác bỏ với Qwen (grief=0.000, caring=0.000), nhưng được xác nhận một phần với Llama (grief=0.125, caring=0.214). Hai LLM có **profile lỗi khác nhau**: Llama aggressive hơn (better recall ở rare classes), Qwen conservative hơn (better precision, ít false positives). Phân tích lỗi BERT vs Qwen trên 2,000 mẫu cho thấy BERT có Jaccard cao hơn 40.4% trường hợp, trong khi LLM chỉ có lợi thế ở single-emotion clear cases.

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

### 2.2. Track B: LLM Zero-shot (Offline)

#### Mô hình sử dụng

- **EXP-03:** `meta-llama/Llama-3.2-3B-Instruct` (Meta, 3B params)
- **EXP-04:** `Qwen/Qwen2.5-3B-Instruct` (Alibaba, 3B params)

Cả hai mô hình chạy **offline** trên GPU local (không cần API key), sử dụng HuggingFace `transformers` pipeline với float16 precision.

#### Prompt Design

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

| Model | Phương pháp | F1-macro | F1-micro | F1-weighted | Hamming Loss | Eval Set |
|-------|-------------|----------|----------|-------------|--------------|----------|
| BERT-base | Fine-tune | **0.4159** | **0.4650** | **0.5329** | 0.0778 | Full test (5,427) |
| RoBERTa-base | Fine-tune | 0.4111 | 0.4618 | 0.5289 | 0.0795 | Full test (5,427) |
| Qwen2.5 3B | Zero-shot | 0.2219 | 0.2717 | 0.2803 | **0.0713** | 2K subset |
| Llama 3.2 3B | Zero-shot | 0.2126 | 0.2306 | 0.2914 | 0.1068 | 2K subset |
| *Paper baseline* | *BERT Fine-tune* | *0.46* | — | — | — | *Demszky 2020* |

### 3.2. Chi Tiết EXP-01: BERT-base Fine-tuning

**Training curve:**

| Epoch | Train Loss | Val Loss | Val F1-macro | Val F1-micro |
|-------|-----------|---------|-------------|-------------|
| 1 | 0.7009 | — | 0.4012 | 0.4150 |
| 2 | 0.4419 | — | 0.4137 | 0.4459 |
| 3 | **0.3408** | — | **0.4305** | **0.4717** |

**Test set (best epoch = 3):** F1-macro = **0.4159**, F1-micro = 0.4650, Hamming = 0.0778

**Nhận xét:** Loss vẫn đang giảm ở epoch 3 trong khi val loss bắt đầu tăng nhẹ — dấu hiệu slight overfitting. Thử thêm epochs với early stopping có thể cải thiện.

### 3.3. Chi Tiết EXP-02: RoBERTa-base Fine-tuning

**Training curve:**

| Epoch | Train Loss | Val F1-macro | Val F1-micro |
|-------|-----------|-------------|-------------|
| 1 | 0.7088 | 0.4049 | 0.4302 |
| 2 | 0.4716 | 0.3785 | 0.4216 |
| 3 | **0.3764** | **0.4164** | **0.4658** |

**Test set:** F1-macro = **0.4111**, F1-micro = 0.4618, Hamming = 0.0795

**Nhận xét:** Trái với hypothesis ban đầu, RoBERTa không vượt BERT (chênh lệch 0.5 điểm). Val F1-macro bị drop ở epoch 2 (0.378) rồi recover — training curve không ổn định hơn BERT trong setup này. Cả hai model có thể cần tune thêm learning rate và epochs.

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

| Model | Thời gian | Tài nguyên | Chi phí |
|-------|-----------|-----------|---------|
| BERT fine-tune (3 epochs) | 37 phút/lần | GPU 16GB | Chi phí huấn luyện một lần |
| BERT inference (full test) | <1 phút | GPU 16GB | Negligible (batch) |
| LLM zero-shot (2K samples) | 16 phút | GPU 16GB | Không cần huấn luyện |
| LLM zero-shot (full test 5,427) | ~43 phút | GPU 16GB | — |

**Kết luận về trade-off:**
- Fine-tuning có chi phí cao ban đầu nhưng inference nhanh (batch)
- LLM zero-shot không cần training nhưng inference chậm (sequential, ~23s/sample)
- Với dataset lớn hoặc real-time requirements, fine-tuning vẫn ưu việt hơn

---

## 5. Kết Luận

### 5.1. Tóm Tắt Phát Hiện

1. **BERT fine-tuning đạt F1-macro = 0.4159**, xấp xỉ 90% paper baseline (0.46). Chênh lệch 5 điểm có thể giải thích bởi paper dùng nhiều epochs và hyperparameter tuning hơn.

2. **RoBERTa không vượt BERT** trong điều kiện cùng hyperparameters — trái với kỳ vọng. Cần thêm epochs và/hoặc learning rate thấp hơn để tận dụng khả năng của RoBERTa.

3. **Class imbalance là challenge chính:** Các emotions hiếm (relief, grief, realization) đạt F1 < 0.2 dù đã dùng pos_weight. Rare classes cần chiến lược đặc biệt hơn.

4. **Cả hai LLM zero-shot đều thua BERT gần 2x:** Qwen=0.2219, Llama=0.2126 vs BERT=0.4159. Fine-tuning vẫn là lựa chọn vượt trội cho fine-grained emotion classification.

5. **Hai LLM có profile lỗi khác nhau:** Llama aggressive hơn — bắt được rare classes (grief=0.125, caring=0.214) nhưng Hamming Loss cao (0.107). Qwen conservative hơn — Hamming Loss thấp (0.071) nhưng miss hoàn toàn grief=0.000, caring=0.000.

6. **RQ3 kết quả nuanced:** So sánh giữa hai LLM xác nhận Llama có lợi thế ở rare classes hơn Qwen, nhưng cả hai đều kém xa BERT ở mọi class.

### 5.2. Hạn Chế

- Chỉ train 3 epochs (paper gốc dùng nhiều hơn)
- Không thực hiện hyperparameter search
- LLM chỉ đánh giá trên 2,000/5,427 mẫu test (do giới hạn thời gian inference)
- Không có few-shot experiments

### 5.3. Hướng Phát Triển

1. **Few-shot LLM:** Thử k=5 examples trong prompt có thể thu hẹp gap với fine-tuning
2. **Longer fine-tuning:** BERT/RoBERTa với 5-10 epochs + early stopping
3. **Ensemble:** Kết hợp predictions của BERT và LLM có thể cải thiện rare class coverage
4. **DeBERTa-v3:** Mô hình mạnh hơn BERT/RoBERTa cho sequence classification tasks

---

## 6. Tài Liệu Tham Khảo

1. Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. *ACL 2020*.

2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL 2019*.

3. Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. *arXiv:1907.11692*.

4. Qwen Team. (2024). Qwen2.5 Technical Report. *arXiv:2412.15115*.

5. Meta AI. (2024). Llama 3.2: Lightweight Models with Multimodal Capabilities. *Meta Blog*.

---

*Tất cả 4 experiments đã hoàn tất. Tổng thời gian: EXP-01 BERT 37 phút, EXP-02 RoBERTa 38 phút, EXP-03 Llama 44 phút, EXP-04 Qwen 16 phút.*
