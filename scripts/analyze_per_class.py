"""
============================================================
analyze_per_class.py — Per-class F1 comparison across models
============================================================

Usage:
    python scripts/analyze_per_class.py

What it does:
    1. Loads per-class F1 from all available *_metrics.json files
    2. Prints a ranked table per emotion
    3. Saves heatmap → results/plots/per_class_f1_heatmap.png
    4. Saves top/bottom emotion summary to stdout
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS_DIR = ROOT / "results" / "metrics"
PLOTS_DIR = ROOT / "results" / "plots"

DISPLAY_MAP = {
    "bert_base_baseline": "BERT-base\n(FT)",
    "roberta_base_baseline": "RoBERTa-base\n(FT)",
    "llama_zeroshot": "Llama 3.2 3B\n(ZS)",
    "qwen_zeroshot": "Qwen2.5 3B\n(ZS)",
}


def load_per_class(metrics_dir: Path) -> dict[str, dict[str, float]]:
    """Return {experiment_name: {emotion: f1}} for all available files.

    Handles two formats:
    - LLM format: top-level "per_class_f1" and "n_samples"
    - Train format: "test_per_class_f1" nested, full test set
    """
    data: dict[str, dict[str, float]] = {}
    seen: set[str] = set()
    for fpath in sorted(metrics_dir.glob("*.json")):
        with open(fpath, "r", encoding="utf-8") as f:
            d = json.load(f)
        exp = d.get("experiment", fpath.stem.replace("_metrics", ""))
        if exp not in DISPLAY_MAP or exp in seen:
            continue

        # LLM format
        if "per_class_f1" in d:
            n = d.get("n_samples", 0)
            if isinstance(n, int) and n < 100:
                continue
            data[exp] = d["per_class_f1"]
            seen.add(exp)
        # Fine-tune format
        elif "test_per_class_f1" in d:
            data[exp] = d["test_per_class_f1"]
            seen.add(exp)

    return data


def print_summary(df: pd.DataFrame) -> None:
    """Print top-5 and bottom-5 emotions averaged across all models."""
    avg = df.mean(axis=1).sort_values(ascending=False)
    print("\n== Top 5 emotions (avg F1 across models) ==")
    for emo, val in avg.head(5).items():
        row = "  ".join(f"{df.loc[emo, c]:.3f}" for c in df.columns)
        print(f"  {emo:<20} avg={val:.3f}  [{row}]")
    print("\n== Bottom 5 emotions (avg F1 across models) ==")
    for emo, val in avg.tail(5).items():
        row = "  ".join(f"{df.loc[emo, c]:.3f}" for c in df.columns)
        print(f"  {emo:<20} avg={val:.3f}  [{row}]")


def save_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(df.columns) * 2.2), 10))
    im = ax.imshow(df.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, fontsize=10)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=9)

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.iloc[i, j]
            color = "white" if val < 0.35 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="F1 score", fraction=0.03, pad=0.02)
    ax.set_title("Per-class F1 by Model (GoEmotions)", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved → {out_path}")


def main() -> None:
    data = load_per_class(METRICS_DIR)
    if not data:
        print("No per-class metrics found. Run experiments first.")
        return

    # Build DataFrame: rows=emotions, cols=models
    all_emotions = list(next(iter(data.values())).keys())
    cols = {DISPLAY_MAP[exp]: data[exp] for exp in data}
    df = pd.DataFrame(cols, index=all_emotions)

    # Sort emotions by avg F1 descending
    df = df.loc[df.mean(axis=1).sort_values(ascending=False).index]

    print_summary(df)
    save_heatmap(df, PLOTS_DIR / "per_class_f1_heatmap.png")

    csv_path = METRICS_DIR / "per_class_comparison.csv"
    df.to_csv(csv_path)
    print(f"Per-class CSV saved → {csv_path}")


if __name__ == "__main__":
    main()
