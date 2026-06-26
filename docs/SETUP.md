# 🚀 Setup Guide

Hướng dẫn cài đặt từ A → Z. Theo thứ tự, đừng skip.

---

## 1. Yêu cầu

- **OS:** Windows 10/11, macOS, hoặc Linux
- **Python:** 3.10+
- **GPU:** NVIDIA với CUDA 12.x (khuyến nghị, không bắt buộc)
- **Anaconda** hoặc Miniconda
- **Git**

---

## 2. Clone Repo

```bash
git clone https://github.com/duyanh0208/goemotions-emotion-classification.git
cd goemotions-emotion-classification
```

---

## 3. Tạo Conda Environment

```bash
conda create -n goemotions python=3.10 -y
conda activate goemotions
```

---

## 4. Cài PyTorch với CUDA

### Cho NVIDIA GPU (CUDA 12.x)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

→ Kỳ vọng:
```
2.x.x
True
```

Nếu `False`: driver NVIDIA chưa đúng version. Update driver hoặc cài CPU-only:
```bash
pip install torch torchvision
```

---

## 5. Cài Dependencies

```bash
pip install -r requirements.txt
```

Mất ~3-5 phút.

---

## 6. Setup LLM Models (Offline)

Track B dùng Llama và Qwen chạy **offline trên local GPU** — không cần API key, không cần billing.

### 6.1. Tải model về máy (lần đầu cần internet)

Model sẽ tự tải về HuggingFace cache khi chạy inference lần đầu:

```bash
# Llama 3.2 3B (~6 GB) — tải tự động
python -m src.llm_inference --config configs/llama_zeroshot.yaml --n_samples 5

# Qwen2.5 3B (~6 GB) — tải tự động
python -m src.llm_inference --config configs/qwen_zeroshot.yaml --n_samples 5
```

Sau khi tải xong, bật `local_files_only: true` trong YAML để chạy hoàn toàn offline.

### 6.2. Yêu cầu VRAM

| Setup | VRAM tối thiểu | Ghi chú |
|-------|----------------|---------|
| float16 (mặc định) | ~7–8 GB | Máy trường RTX 2000 Ada (16 GB) ✓ |
| 4-bit quantization | ~2–3 GB | Bật `load_in_4bit: true` trong YAML, cần `bitsandbytes` |
| CPU-only | RAM ~12 GB | Rất chậm (~1 phút/sample) |

Để bật 4-bit trên máy nhà (4 GB VRAM):
```yaml
# Trong configs/llama_zeroshot.yaml hoặc qwen_zeroshot.yaml
model:
  load_in_4bit: true
```

Cài thêm:
```bash
pip install bitsandbytes
```

### 6.3. W&B API Key (free, cho theo dõi training)

```bash
cp .env.example .env
```

1. Đăng ký tại https://wandb.ai/signup
2. Vào https://wandb.ai/authorize
3. Copy API key
4. Paste vào `.env`:
   ```
   WANDB_API_KEY=abc123...
   ```

Hoặc đăng nhập trực tiếp:
```bash
wandb login
# Paste API key khi được hỏi
```

---

## 7. Smoke Test

### 7.1. Run EDA
```bash
python scripts/run_eda.py
```

→ Kỳ vọng: in ra stats dataset, save vào `results/eda/`.

### 7.2. Run Training (debug mode)
```bash
python -m src.train --config configs/bert_base_debug.yaml
```

→ Chạy ~5-10 phút. Verify pipeline OK.

Nếu lỗi `CUDA out of memory`:
```bash
# Giảm batch size
python -m src.train --config configs/bert_base_debug.yaml --batch_size 4
```

---

## 8. Full Training

### Trên máy có GPU (16GB+)
```bash
python -m src.train --config configs/bert_base.yaml
```

→ Chạy ~1-1.5h trên RTX 2000 Ada.

### Trên máy GPU nhỏ (4GB)
```bash
python -m src.train --config configs/bert_base.yaml --batch_size 8
```

→ Chạy ~3-4h trên GTX 1650 Ti.

### Theo dõi real-time
- Mở https://wandb.ai → project `goemotions-emotion-classification`
- Xem loss curves, F1 metrics, GPU usage

---

## 9. Troubleshooting

### Lỗi: `ModuleNotFoundError: No module named 'src'`
→ Phải chạy từ root folder repo:
```bash
cd goemotions-emotion-classification
python -m src.train --config ...
```

### Lỗi: `huggingface_hub` symlink warning trên Windows
→ Bỏ qua, không ảnh hưởng. Hoặc bật Developer Mode trong Windows Settings.

### Lỗi: `wandb login` không hiện prompt paste
→ Set env variable thay vì login:
```bash
# Trong PowerShell:
$env:WANDB_API_KEY="abc123..."

# Trong CMD:
set WANDB_API_KEY=abc123...
```

### Training quá chậm
- Verify GPU đang được dùng: log đầu sẽ in `Using GPU: ...`
- Tăng `num_workers` trong config (`data.num_workers: 4`)
- Dùng máy mạnh hơn

---

## 10. Next Steps

Sau khi setup xong:
1. Đọc [PROJECT_PLAN.md](PROJECT_PLAN.md) để hiểu roadmap
2. Chạy experiments theo thứ tự EXP-01 → EXP-09 trong [EXPERIMENTS.md](EXPERIMENTS.md)
3. Xem kết quả đầy đủ trong [EXPERIMENTS.md](EXPERIMENTS.md)
