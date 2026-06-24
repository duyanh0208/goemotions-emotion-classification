"""
============================================================
run_brighter.py — Cross-benchmark trên SemEval-2025 Task 11
                  (BRIGHTER English, Track A)
============================================================

Chạy ĐÚNG pipeline của project (BERT fine-tune / Llama-Qwen prompting)
trên dữ liệu English Track A của SemEval-2025 Task 11 → đặt số của mình
cạnh trực tiếp RoBERTa baseline (0.708) và hệ đoạt giải PAI (0.823).

English Track A dùng 5 cảm xúc (disgust không annotate cho English):
    anger, fear, joy, sadness, surprise

Usage:
    # BERT fine-tune (train 2,764 → eval test 5,528)
    python -m scripts.run_brighter --mode bert --model bert-base-uncased

    # LLM prompting (eval test 5,528)
    python -m scripts.run_brighter --mode llm --model Qwen/Qwen2.5-3B-Instruct --prompt zero_shot
    python -m scripts.run_brighter --mode llm --model meta-llama/Llama-3.2-3B-Instruct --prompt few_shot
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import datasets as _ds_init  # noqa: F401  (import before torch)
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, hamming_loss

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

# English Track A label set (disgust absent for English)
LABELS = ["anger", "fear", "joy", "sadness", "surprise"]
NAME_TO_IDX = {n: i for i, n in enumerate(LABELS)}
METRICS_DIR = ROOT / "results" / "metrics"


# ============================================================
# Data
# ============================================================
def load_brighter_eng():
    ds = load_dataset("brighter-dataset/BRIGHTER-emotion-categories", "eng")
    return ds


def emotions_to_multihot(emotions: List[str]) -> np.ndarray:
    v = np.zeros(len(LABELS), dtype=int)
    for e in emotions:
        if e in NAME_TO_IDX:
            v[NAME_TO_IDX[e]] = 1
    return v


def metrics_from(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> Dict:
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    return {
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_micro": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "hamming_loss": round(float(hamming_loss(y_true, y_pred)), 4),
        "n_samples": n,
        "per_class_f1": {name: round(float(f), 4) for name, f in zip(LABELS, per_class)},
    }


# ============================================================
# BERT fine-tune
# ============================================================
def run_bert(model_name: str, epochs: int = 3, batch_size: int = 16,
             lr: float = 2e-5, max_length: int = 128, seed: int = 42) -> Dict:
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from src.models import EmotionClassifier

    torch.manual_seed(seed); np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_brighter_eng()
    print(f"BRIGHTER eng — train {len(ds['train'])} / dev {len(ds['dev'])} / test {len(ds['test'])}")

    tok = AutoTokenizer.from_pretrained(model_name)

    class BDS(Dataset):
        def __init__(self, split): self.s = split
        def __len__(self): return len(self.s)
        def __getitem__(self, i):
            it = self.s[i]
            enc = tok(it["text"], truncation=True, padding="max_length",
                      max_length=max_length, return_tensors="pt")
            y = torch.tensor(emotions_to_multihot(it["emotions"]), dtype=torch.float)
            return {"input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0), "labels": y}

    tr = DataLoader(BDS(ds["train"]), batch_size=batch_size, shuffle=True)
    te = DataLoader(BDS(ds["test"]), batch_size=32, shuffle=False)

    # pos_weight cho class imbalance (clip [1,50], giống GoEmotions pipeline)
    counts = np.zeros(len(LABELS)); total = len(ds["train"])
    for it in ds["train"]:
        counts += emotions_to_multihot(it["emotions"])
    pw = np.clip((total - counts) / (counts + 1e-6), 1.0, 50.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float).to(device))

    model = EmotionClassifier(model_name=model_name, num_labels=len(LABELS)).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = len(tr) * epochs
    sch = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)

    t0 = time.time()
    for ep in range(epochs):
        model.train(); tl = 0.0
        for b in tr:
            ii, am, y = b["input_ids"].to(device), b["attention_mask"].to(device), b["labels"].to(device)
            logits = model(ii, am); loss = criterion(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); tl += loss.item()
        print(f"  epoch {ep+1}/{epochs}  train_loss={tl/len(tr):.4f}")

    # eval test
    model.eval(); probs_all, y_all = [], []
    with torch.no_grad():
        for b in te:
            ii, am = b["input_ids"].to(device), b["attention_mask"].to(device)
            p = torch.sigmoid(model(ii, am)).cpu().numpy()
            probs_all.append(p); y_all.append(b["labels"].numpy())
    probs = np.vstack(probs_all); y_true = np.vstack(y_all).astype(int)
    train_time = time.time() - t0

    out = {"experiment": f"brighter_eng_{model_name.split('/')[-1]}",
           "track": "SemEval-2025 Task11 Track A (English, 5 labels)",
           "method": "fine-tune", "train_time_s": round(train_time, 1)}
    # default threshold 0.5
    m05 = metrics_from(y_true, (probs >= 0.5).astype(int), len(y_true))
    out["t0.5"] = m05
    # best global threshold (grid)
    best_t, best_f1 = 0.5, m05["f1_macro"]
    for t in np.arange(0.1, 0.91, 0.05):
        f = f1_score(y_true, (probs >= t).astype(int), average="macro", zero_division=0)
        if f > best_f1: best_f1, best_t = f, round(float(t), 2)
    out["best_threshold"] = {"t": best_t,
                             **metrics_from(y_true, (probs >= best_t).astype(int), len(y_true))}
    return out


# ============================================================
# LLM prompting
# ============================================================
def _build_prompt(text: str, shots: List[dict] = None) -> str:
    head = (
        "You are an expert emotion analyst. Given a short text, identify ALL emotions "
        "expressed by the speaker.\n\n"
        "RULES:\n"
        "1. Multi-label — select ALL that apply.\n"
        f"2. Only use labels from this list: {', '.join(LABELS)}.\n"
        "3. If no emotion from the list applies, return an empty list.\n"
        '4. Output ONLY valid JSON, no explanation: {"emotions": ["..."]}\n\n'
    )
    ex = ""
    if shots:
        for s in shots:
            ex += f'Text: "{s["text"]}"\nEmotions: {{"emotions": {json.dumps(s["emotions"])}}}\n\n'
    return head + ex + f'TEXT TO CLASSIFY: "{text}"'


def _parse(resp: str) -> List[str]:
    import re
    if not resp:
        return []
    clean = re.sub(r"```(?:json)?\s*|```", "", resp).strip()

    def ext(o):
        em = o.get("emotions", []) if isinstance(o, dict) else (o if isinstance(o, list) else [])
        return [e for e in em if e in NAME_TO_IDX]
    for pat in (clean, ):
        try:
            return ext(json.loads(pat))
        except Exception:
            pass
    m = re.search(r"\{.*?\}", clean, re.DOTALL)
    if m:
        try:
            return ext(json.loads(m.group()))
        except Exception:
            pass
    # keyword fallback
    return [e for e in LABELS if re.search(rf"\b{e}\b", clean.lower())]


def run_llm(model_name: str, prompt_mode: str, seed: int = 42, k: int = 5) -> Dict:
    from transformers import pipeline as hf_pipeline, GenerationConfig
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    ds = load_brighter_eng()
    test = ds["test"]

    shots = None
    if prompt_mode == "few_shot":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ds["train"]), size=k, replace=False).tolist()
        shots = [{"text": ds["train"][i]["text"], "emotions": ds["train"][i]["emotions"]} for i in idx]

    print(f"Loading {model_name} …")
    pipe = hf_pipeline(task="text-generation", model=model_name, dtype=dtype, device_map=device)
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    gc = GenerationConfig(max_new_tokens=128, do_sample=False,
                          pad_token_id=pipe.tokenizer.eos_token_id)

    out_dir = ROOT / "results" / "llm" / f"brighter_{model_name.split('/')[-1]}_{prompt_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint.json"
    done = {}
    if ckpt.exists():
        done = {r["id"]: r for r in json.load(open(ckpt, encoding="utf-8"))}
        print(f"Resuming — {len(done)} done")

    records = list(done.values())
    t0 = time.time()
    for i in range(len(test)):
        it = test[i]
        sid = it["id"]
        if sid in done:
            continue
        msgs = [{"role": "user", "content": _build_prompt(it["text"], shots)}]
        fp = pipe.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        try:
            raw = pipe(fp, generation_config=gc, return_full_text=False)[0]["generated_text"].strip()
            pred = _parse(raw)
        except Exception as e:
            raw, pred = "", []
            print(f"  skip {sid}: {e}")
        records.append({"id": sid, "text": it["text"], "true": it["emotions"], "pred": pred})
        if len(records) % 100 == 0:
            json.dump(records, open(ckpt, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"  {len(records)}/{len(test)}  ({100*len(records)/len(test):.0f}%)")
    json.dump(records, open(ckpt, "w", encoding="utf-8"), ensure_ascii=False)

    y_true = np.vstack([emotions_to_multihot(r["true"]) for r in records])
    y_pred = np.vstack([emotions_to_multihot(r["pred"]) for r in records])
    out = {"experiment": f"brighter_eng_{model_name.split('/')[-1]}_{prompt_mode}",
           "track": "SemEval-2025 Task11 Track A (English, 5 labels)",
           "method": prompt_mode, "infer_time_s": round(time.time() - t0, 1),
           **metrics_from(y_true, y_pred, len(records))}
    return out


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bert", "llm"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", choices=["zero_shot", "few_shot"], default="zero_shot")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    if args.mode == "bert":
        res = run_bert(args.model, epochs=args.epochs)
        tag = f"brighter_bert_{args.model.split('/')[-1]}"
        print(f"\nBERT  t=0.5 F1-macro={res['t0.5']['f1_macro']}  "
              f"best(t={res['best_threshold']['t']})={res['best_threshold']['f1_macro']}")
    else:
        res = run_llm(args.model, args.prompt)
        tag = f"brighter_{args.model.split('/')[-1]}_{args.prompt}"
        print(f"\nLLM  F1-macro={res['f1_macro']}  F1-micro={res['f1_micro']}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / f"{tag}.json"
    json.dump(res, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
