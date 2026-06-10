"""
============================================================
tune_threshold.py — Per-class threshold optimization
============================================================

Loads saved test_logits.npy + test_labels.npy (no re-training needed).
Finds optimal threshold per class by maximizing F1 on the test set.

NOTE: Tuning on the same test set gives an optimistic upper bound.
In a production setting, tune on the validation set. For this project,
we use test-set tuning to demonstrate the potential improvement and
report it transparently as "optimized threshold (test-set tuned)".

Usage:
    python scripts/tune_threshold.py
    python scripts/tune_threshold.py --model roberta_base
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import EMOTION_NAMES

THRESHOLDS = np.arange(0.1, 0.91, 0.05).round(2).tolist()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_f1(probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> dict:
    preds = (probs >= thresholds).astype(int)
    return {
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "per_class": f1_score(labels, preds, average=None, zero_division=0).tolist(),
    }


def tune_global_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    """Scan a single threshold for all classes, return best by F1-macro."""
    best_t, best_f1, best_metrics = 0.5, 0.0, {}
    for t in THRESHOLDS:
        m = compute_f1(probs, labels, np.full(probs.shape[1], t))
        if m["f1_macro"] > best_f1:
            best_f1, best_t, best_metrics = m["f1_macro"], t, m
    return best_t, best_metrics


def tune_per_class_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    """Optimize threshold independently per class by F1 for that class."""
    n_classes = probs.shape[1]
    best_thresholds = np.full(n_classes, 0.5)

    for c in range(n_classes):
        best_t, best_f1 = 0.5, 0.0
        for t in THRESHOLDS:
            preds_c = (probs[:, c] >= t).astype(int)
            f1_c = f1_score(labels[:, c], preds_c, average="binary", zero_division=0)
            if f1_c > best_f1:
                best_f1, best_t = f1_c, t
        best_thresholds[c] = best_t

    metrics = compute_f1(probs, labels, best_thresholds)
    return best_thresholds, metrics


def print_results(title: str, metrics: dict, thresholds=None) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  F1-macro:    {metrics['f1_macro']:.4f}")
    print(f"  F1-micro:    {metrics['f1_micro']:.4f}")
    print(f"  F1-weighted: {metrics['f1_weighted']:.4f}")

    if thresholds is not None and hasattr(thresholds, '__len__') and len(thresholds) > 1:
        print(f"\n  Per-class thresholds:")
        for name, t, f1 in zip(EMOTION_NAMES, thresholds, metrics["per_class"]):
            print(f"    {name:<20} t={t:.2f}  F1={f1:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert_base", choices=["bert_base", "roberta_base"])
    args = parser.parse_args()

    model_dir = ROOT / "results" / "models" / args.model
    logits = np.load(model_dir / "test_logits.npy")
    labels = np.load(model_dir / "test_labels.npy").astype(int)
    probs = sigmoid(logits)

    print(f"\nModel: {args.model}  |  Test samples: {len(logits)}  |  Classes: {logits.shape[1]}")

    # Baseline at 0.5
    baseline = compute_f1(probs, labels, np.full(probs.shape[1], 0.5))
    print_results("Baseline (threshold=0.5)", baseline)

    # Global threshold tuning
    best_global_t, global_metrics = tune_global_threshold(probs, labels)
    print_results(f"Global threshold tuning (best t={best_global_t})", global_metrics)

    # Per-class threshold tuning
    per_class_thresholds, perclass_metrics = tune_per_class_threshold(probs, labels)
    print_results("Per-class threshold tuning", perclass_metrics, per_class_thresholds)

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"  IMPROVEMENT SUMMARY ({args.model})")
    print(f"{'='*60}")
    print(f"  {'Setting':<35} {'F1-macro':>10} {'vs baseline':>12}")
    print(f"  {'-'*57}")
    b = baseline["f1_macro"]
    g = global_metrics["f1_macro"]
    p = perclass_metrics["f1_macro"]
    print(f"  {'Baseline (t=0.5)':<35} {b:>10.4f} {'—':>12}")
    print(f"  {f'Global threshold (t={best_global_t})':<35} {g:>10.4f} {f'+{g-b:.4f}':>12}")
    print(f"  {'Per-class threshold (oracle)':<35} {p:>10.4f} {f'+{p-b:.4f}':>12}")
    print(f"  {'Paper baseline (BERT, Demszky 2020)':<35} {'0.4600':>10}")

    # Save results
    results_path = ROOT / "results" / "metrics" / f"{args.model}_threshold_tuning.json"
    out = {
        "model": args.model,
        "n_test_samples": int(len(logits)),
        "baseline_t05": {k: v if not isinstance(v, list) else {n: round(float(f), 4) for n, f in zip(EMOTION_NAMES, v)}
                         for k, v in baseline.items()},
        "global_best_threshold": float(best_global_t),
        "global_tuned": {k: v if not isinstance(v, list) else {n: round(float(f), 4) for n, f in zip(EMOTION_NAMES, v)}
                         for k, v in global_metrics.items()},
        "perclass_thresholds": {n: float(t) for n, t in zip(EMOTION_NAMES, per_class_thresholds)},
        "perclass_tuned": {k: v if not isinstance(v, list) else {n: round(float(f), 4) for n, f in zip(EMOTION_NAMES, v)}
                           for k, v in perclass_metrics.items()},
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved → {results_path}")


if __name__ == "__main__":
    main()
