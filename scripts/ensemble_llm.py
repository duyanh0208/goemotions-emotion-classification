"""
============================================================
EXP-08 — LLM Ensemble (mô phỏng PAI, SemEval-2025 Task 11)
============================================================

Kết hợp predictions của các LLM đã chạy (Llama + Qwen, zero/few-shot)
trên cùng test set 5,427 mẫu bằng các chiến lược voting:

    - union          : nhãn được dự đoán bởi BẤT KỲ model nào (recall ↑)
    - intersection   : nhãn được dự đoán bởi TẤT CẢ model (precision ↑)
    - majority (k≥t) : nhãn được ≥ t model bình chọn (cân bằng)

Không cần GPU — chỉ xử lý JSON predictions đã có.

Usage:
    python -m scripts.ensemble_llm
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import f1_score, hamming_loss

from src.data import EMOTION_NAMES

RESULTS_DIR = Path("results")
PRED_DIR = RESULTS_DIR / "llm"
METRICS_DIR = RESULTS_DIR / "metrics"
NAME_TO_IDX = {n: i for i, n in enumerate(EMOTION_NAMES)}


def _multihot(labels: List[str]) -> np.ndarray:
    v = np.zeros(len(EMOTION_NAMES), dtype=int)
    for e in labels:
        if e in NAME_TO_IDX:
            v[NAME_TO_IDX[e]] = 1
    return v


def _load_preds(name: str) -> Dict[str, dict]:
    """Load predictions.json → {sample_id: record}."""
    path = PRED_DIR / name / "predictions.json"
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {r["id"]: r for r in records}


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_micro": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "hamming_loss": round(float(hamming_loss(y_true, y_pred)), 4),
    }


def ensemble(member_names: List[str], strategy: str, vote_threshold: int = 2) -> dict:
    """
    Combine member predictions.

    strategy: "union" | "intersection" | "majority"
    vote_threshold: dùng cho "majority" (số phiếu tối thiểu để bật nhãn).
    """
    members = [_load_preds(n) for n in member_names]
    # ids chung (theo member đầu, các member khác cùng seed nên trùng)
    ids = list(members[0].keys())

    y_true, y_pred = [], []
    for sid in ids:
        true_vec = _multihot(members[0][sid]["true_labels"])
        votes = np.zeros(len(EMOTION_NAMES), dtype=int)
        for m in members:
            votes += _multihot(m[sid]["predicted_labels"])

        if strategy == "union":
            pred_vec = (votes >= 1).astype(int)
        elif strategy == "intersection":
            pred_vec = (votes >= len(members)).astype(int)
        elif strategy == "majority":
            pred_vec = (votes >= vote_threshold).astype(int)
        else:
            raise ValueError(strategy)

        # tránh vector rỗng → fallback neutral (giống pipeline gốc)
        if pred_vec.sum() == 0:
            pred_vec[NAME_TO_IDX["neutral"]] = 1

        y_true.append(true_vec)
        y_pred.append(pred_vec)

    m = _metrics(np.array(y_true), np.array(y_pred))
    m["n_samples"] = len(ids)
    return m


def main() -> None:
    # Baselines đơn lẻ (để so sánh)
    print("=" * 78)
    print("EXP-08  LLM Ensemble trên 5,427 mẫu (Llama + Qwen)")
    print("=" * 78)

    configs = [
        # (label, member_names, strategy, threshold)
        ("Qwen FS (single)",            ["qwen_fewshot"],                                 "union", 1),
        ("Llama FS (single)",           ["llama_fewshot"],                                "union", 1),
        ("FS  union   (Llama∪Qwen)",    ["llama_fewshot", "qwen_fewshot"],                "union", 1),
        ("FS  inter   (Llama∩Qwen)",    ["llama_fewshot", "qwen_fewshot"],                "intersection", 2),
        ("All4 union",                  ["llama_fewshot", "qwen_fewshot",
                                         "llama_zeroshot", "qwen_zeroshot"],              "union", 1),
        ("All4 majority (≥2)",          ["llama_fewshot", "qwen_fewshot",
                                         "llama_zeroshot", "qwen_zeroshot"],              "majority", 2),
        ("All4 majority (≥3)",          ["llama_fewshot", "qwen_fewshot",
                                         "llama_zeroshot", "qwen_zeroshot"],              "majority", 3),
    ]

    print(f"{'Strategy':28} {'F1-macro':>9} {'F1-micro':>9} {'F1-weighted':>11} {'Hamming':>8}")
    print("-" * 78)
    results = {}
    best = None
    for label, members, strat, thr in configs:
        m = ensemble(members, strat, thr)
        results[label] = {"members": members, "strategy": strat, "vote_threshold": thr, **m}
        print(f"{label:28} {m['f1_macro']:>9.4f} {m['f1_micro']:>9.4f} "
              f"{m['f1_weighted']:>11.4f} {m['hamming_loss']:>8.4f}")
        if best is None or m["f1_macro"] > best[1]["f1_macro"]:
            best = (label, results[label])

    print("-" * 78)
    print(f"BEST ensemble: {best[0]}  →  F1-macro = {best[1]['f1_macro']:.4f}")

    out = {
        "experiment": "ensemble_llm",
        "description": "EXP-08 LLM ensemble (Llama+Qwen voting) on full 5427 test",
        "best_strategy": best[0],
        "best": best[1],
        "all_strategies": results,
    }
    out_path = METRICS_DIR / "ensemble_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
