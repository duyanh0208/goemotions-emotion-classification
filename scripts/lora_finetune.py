"""
============================================================
lora_finetune.py — LoRA fine-tune một LLM (Qwen2.5-3B) làm
                   multi-label emotion classifier (kiểu PAI)
============================================================

Mục đích: ĐÓNG VÒNG lập luận. Track LLM hiện tại chỉ prompt-only;
script này tự chứng minh "fine-tune LLM mạnh hơn prompt-only" bằng cách
LoRA fine-tune chính Qwen2.5-3B (LLM-as-classifier, giống ý tưởng PAI:
LLM embedding + classification head) rồi so với:
  - Qwen prompt-only (zero/few-shot)
  - BERT fine-tune
  - (BRIGHTER) baseline 0.708 / PAI 0.823

Usage:
    python -m scripts.lora_finetune --dataset brighter   --epochs 5
    python -m scripts.lora_finetune --dataset goemotions --epochs 2
"""

import argparse
import json
import time
from pathlib import Path

import datasets as _ds_init  # noqa: F401  (before torch)
import numpy as np
import torch
from sklearn.metrics import f1_score, hamming_loss

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
METRICS_DIR = ROOT / "results" / "metrics"
BRIGHTER_LABELS = ["anger", "fear", "joy", "sadness", "surprise"]


def build_data(dataset: str):
    """Return (train_texts, train_Y, test_texts, test_Y, label_names)."""
    if dataset == "brighter":
        from datasets import load_dataset
        ds = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "eng")
        labels = BRIGHTER_LABELS
        idx = {n: i for i, n in enumerate(labels)}

        def to_y(split):
            Y = np.zeros((len(split), len(labels)), dtype=np.float32)
            for i, em in enumerate(split["emotions"]):
                for e in em:
                    if e in idx:
                        Y[i, idx[e]] = 1.0
            return Y
        return (ds["train"]["text"], to_y(ds["train"]),
                ds["test"]["text"], to_y(ds["test"]), labels)
    else:  # goemotions
        from src.data import EMOTION_NAMES, load_goemotions
        ds = load_goemotions()
        labels = EMOTION_NAMES

        def to_y(split):
            Y = np.zeros((len(split), len(labels)), dtype=np.float32)
            for i, lab in enumerate(split["labels"]):
                for j in lab:
                    Y[i, j] = 1.0
            return Y
        return (ds["train"]["text"], to_y(ds["train"]),
                ds["test"]["text"], to_y(ds["test"]), labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["brighter", "goemotions"], required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=128)
    args = ap.parse_args()

    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)
    from peft import LoraConfig, get_peft_model, TaskType
    import datasets as hfds

    tr_x, tr_y, te_x, te_y, labels = build_data(args.dataset)
    K = len(labels)
    print(f"[{args.dataset}] train={len(tr_x)} test={len(te_x)} labels={K}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=K, problem_type="multi_label_classification",
        torch_dtype=torch.bfloat16,
    )
    model.config.pad_token_id = tok.pad_token_id

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def make_ds(texts, Y):
        d = hfds.Dataset.from_dict({"text": list(texts), "labels": Y.tolist()})
        def tok_fn(b):
            return tok(b["text"], truncation=True, padding="max_length", max_length=args.max_len)
        d = d.map(tok_fn, batched=True, remove_columns=["text"])
        return d

    train_ds = make_ds(tr_x, tr_y)
    test_ds = make_ds(te_x, te_y)

    out_dir = ROOT / "results" / "models" / f"lora_qwen_{args.dataset}"
    targs = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.accum,
        per_device_eval_batch_size=16, learning_rate=args.lr, warmup_ratio=0.05,
        logging_steps=50, save_strategy="no", report_to=[], bf16=True, fp16=False,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds)

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    # Predict on test
    pred = trainer.predict(test_ds)
    logits = pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = 1 / (1 + np.exp(-logits.astype(np.float32)))
    y_pred = (probs >= 0.5).astype(int)
    y_true = te_y.astype(int)

    per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    res = {
        "experiment": f"lora_qwen_{args.dataset}",
        "base_model": BASE_MODEL, "method": "LoRA fine-tune (multi-label classifier)",
        "dataset": args.dataset, "n_test": len(te_x), "epochs": args.epochs,
        "train_time_s": round(train_time, 1),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_micro": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "hamming_loss": round(float(hamming_loss(y_true, y_pred)), 4),
        "per_class_f1": {n: round(float(f), 4) for n, f in zip(labels, per_class)},
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / f"lora_qwen_{args.dataset}.json"
    json.dump(res, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nLoRA Qwen [{args.dataset}]  F1-macro={res['f1_macro']}  "
          f"F1-micro={res['f1_micro']}  ({train_time/60:.1f} min)")
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
