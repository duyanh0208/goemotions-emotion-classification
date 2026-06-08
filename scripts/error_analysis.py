"""
============================================================
error_analysis.py — BERT vs LLM disagreement analysis
============================================================

Compares BERT predictions against an LLM's predictions and
identifies the most interesting disagreement cases.

Usage:
    python scripts/error_analysis.py
    python scripts/error_analysis.py --llm results/llm/qwen_zeroshot/predictions.json --n 50

Output:
    - Prints top-N disagreement cases with text + labels
    - Saves results/analysis/disagreement_analysis.json
    - Saves results/analysis/error_categories.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# datasets must be imported before torch (avoids CUDA multiprocessing crash on Windows)
from src.data import EMOTION_NAMES, load_goemotions

import numpy as np
import torch

ANALYSIS_DIR = ROOT / "results" / "analysis"
BERT_LOGITS = ROOT / "results" / "models" / "bert_base" / "test_logits.npy"
BERT_LABELS = ROOT / "results" / "models" / "bert_base" / "test_labels.npy"
THRESHOLD = 0.5


# ============================================================
# Helpers
# ============================================================
def load_bert_predictions() -> dict[str, dict] | None:
    """
    Load BERT predictions from saved test_logits.npy.
    Loads the full test split to get sample IDs in order.
    Returns {sample_id: {predicted_labels, true_labels}} or None if missing.
    """
    if not BERT_LOGITS.exists():
        print(f"  [WARN] {BERT_LOGITS} not found — skipping BERT comparison")
        return None

    print("Loading BERT test logits…")
    logits = np.load(BERT_LOGITS)  # (5427, 28)
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds_binary = (probs >= THRESHOLD).astype(int)

    print("Loading GoEmotions test split to get sample IDs…")
    ds = load_goemotions()
    test_ds = ds["test"]

    assert len(test_ds) == len(preds_binary), (
        f"Length mismatch: dataset={len(test_ds)} logits={len(preds_binary)}"
    )

    records: dict[str, dict] = {}
    for idx in range(len(test_ds)):
        sample = test_ds[idx]
        sample_id = sample.get("id", str(idx))
        pred_indices = np.where(preds_binary[idx])[0].tolist()
        true_indices = sample["labels"]
        records[sample_id] = {
            "predicted_labels": [EMOTION_NAMES[i] for i in pred_indices],
            "true_labels": [EMOTION_NAMES[i] for i in true_indices],
        }

    print(f"BERT predictions loaded for {len(records)} test samples.")
    return records


def load_llm_predictions(path: Path) -> dict[str, dict]:
    """Load LLM predictions.json → {sample_id: record}."""
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {r["id"]: r for r in records}


def multihot_to_set(labels: list[str]) -> frozenset[str]:
    return frozenset(labels)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def categorize(true: frozenset, bert_pred: frozenset, llm_pred: frozenset) -> str:
    bert_j = jaccard(true, bert_pred)
    llm_j = jaccard(true, llm_pred)
    if bert_j == 1.0 and llm_j == 1.0:
        return "both_correct"
    if bert_j == 1.0:
        return "bert_wins"
    if llm_j == 1.0:
        return "llm_wins"
    if bert_j > llm_j + 0.2:
        return "bert_closer"
    if llm_j > bert_j + 0.2:
        return "llm_closer"
    return "both_wrong_similar"


# ============================================================
# Main
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm",
        default="results/llm/qwen_zeroshot/predictions.json",
        help="Path to LLM predictions.json",
    )
    parser.add_argument("--n", type=int, default=50, help="Number of top disagreements to save")
    args = parser.parse_args()

    llm_path = ROOT / args.llm
    if not llm_path.exists():
        print(f"LLM predictions not found: {llm_path}")
        print("Run the LLM inference first.")
        sys.exit(1)

    print(f"Loading LLM predictions from {llm_path}…")
    llm_preds = load_llm_predictions(llm_path)

    bert_preds = load_bert_predictions()

    records: list[dict[str, Any]] = []
    categories: dict[str, int] = {}

    for sample_id, llm_rec in llm_preds.items():
        true_set = multihot_to_set(llm_rec["true_labels"])
        llm_set = multihot_to_set(llm_rec["predicted_labels"])

        if bert_preds and sample_id in bert_preds:
            bert_set = multihot_to_set(bert_preds[sample_id]["predicted_labels"])
        else:
            bert_set = frozenset()

        llm_j = jaccard(true_set, llm_set)
        bert_j = jaccard(true_set, bert_set) if bert_set is not None else None
        category = categorize(true_set, bert_set, llm_set) if bert_set else "llm_only"

        categories[category] = categories.get(category, 0) + 1
        disagree = bool(bert_set) and (bert_set != llm_set)

        records.append({
            "id": sample_id,
            "text": llm_rec["text"],
            "true_labels": sorted(true_set),
            "bert_predicted": sorted(bert_set),
            "llm_predicted": sorted(llm_set),
            "bert_jaccard": round(bert_j, 3) if bert_j is not None else None,
            "llm_jaccard": round(llm_j, 3),
            "disagree": disagree,
            "category": category,
        })

    # Sort disagreements by gap size
    disagree_records = [r for r in records if r["disagree"]]
    disagree_records.sort(
        key=lambda r: abs((r["bert_jaccard"] or 0) - r["llm_jaccard"]),
        reverse=True,
    )
    top_n = disagree_records[: args.n]

    # === Print summary ===
    print(f"\n{'='*72}")
    print(f"Total samples: {len(records):,}  |  Disagreements: {len(disagree_records):,}")
    disagree_pct = 100 * len(disagree_records) / len(records) if records else 0
    print(f"Disagreement rate: {disagree_pct:.1f}%")
    print(f"\nCategory breakdown:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(records) if records else 0
        print(f"  {cat:<25} {count:>6}  ({pct:.1f}%)")

    # Find samples where LLM beats BERT (interesting for the report)
    llm_wins = [r for r in records if r["category"] == "llm_wins"]
    bert_wins = [r for r in records if r["category"] == "bert_wins"]
    print(f"\nBERT wins (exact match, LLM wrong): {len(bert_wins)}")
    print(f"LLM  wins (exact match, BERT wrong): {len(llm_wins)}")

    print(f"\n{'='*72}")
    print(f"Top {min(20, len(top_n))} disagreement cases (by |BERT_J - LLM_J|):")
    print(f"{'='*72}")
    for i, r in enumerate(top_n[:20]):
        b_str = f"{r['bert_jaccard']:.3f}" if r["bert_jaccard"] is not None else " N/A"
        print(f"\n[{i+1:2d}] {r['id']}  BERT_J={b_str}  LLM_J={r['llm_jaccard']:.3f}  cat={r['category']}")
        print(f"      Text : {r['text'][:85]}{'…' if len(r['text']) > 85 else ''}")
        print(f"      True : {r['true_labels']}")
        print(f"      BERT : {r['bert_predicted']}")
        print(f"      LLM  : {r['llm_predicted']}")

    # === Save outputs ===
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    out_path = ANALYSIS_DIR / "disagreement_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(top_n, f, indent=2, ensure_ascii=False)
    print(f"\nTop-{args.n} disagreements saved → {out_path}")

    cat_path = ANALYSIS_DIR / "error_categories.json"
    with open(cat_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_llm_samples": len(records),
            "disagreements": len(disagree_records),
            "disagreement_rate_pct": round(disagree_pct, 1),
            "bert_exact_wins": len(bert_wins),
            "llm_exact_wins": len(llm_wins),
            "categories": categories,
        }, f, indent=2)
    print(f"Category summary saved → {cat_path}")

    # Show a few LLM-wins examples (good for the paper)
    if llm_wins:
        print(f"\n=== LLM wins (LLM correct, BERT wrong) — first 5 ===")
        for r in llm_wins[:5]:
            print(f"  [{r['id']}] {r['text'][:70]}")
            print(f"    True={r['true_labels']}  BERT={r['bert_predicted']}  LLM={r['llm_predicted']}")


if __name__ == "__main__":
    main()
