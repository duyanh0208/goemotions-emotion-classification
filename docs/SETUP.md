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
git clone https://github.com/<your-username>/goemotions-emotion-classification.git
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

## 6. Setup API Keys

### 6.1. Tạo file `.env`

```bash
cp .env.example .env
```

### 6.2. Lấy Gemini API Key (free)
1. Vào https://aistudio.google.com/apikey
2. Đăng nhập Google
3. Click "Create API key"
4. Copy key
5. Paste vào `.env`:
   ```
   GEMINI_API_KEY=AIza...
   ```

### 6.3. Lấy W&B API Key (free)
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
2. Đọc [METHODOLOGY.md](METHODOLOGY.md) để hiểu chi tiết phương pháp
3. Chạy experiments theo Week 1 trong PROJECT_PLAN
4. Update [EXPERIMENTS.md](EXPERIMENTS.md) sau mỗi run
