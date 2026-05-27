"""
============================================================
EDA Script — Exploratory Data Analysis cho GoEmotions
============================================================

Run:
    python scripts/run_eda.py

Output:
    - results/eda/label_distribution.csv
    - results/eda/label_names.json
    - results/eda/eda_summary.json
    - results/plots/label_distribution.png (nếu matplotlib có)
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset


def main():
    output_dir = Path("results/eda")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LOADING GoEmotions")
    print("=" * 60)
    ds = load_dataset("google-research-datasets/go_emotions", "simplified")

    label_names = ds["train"].features["labels"].feature.names
    n_labels = len(label_names)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(ds['train']):,}")
    print(f"  Val:   {len(ds['validation']):,}")
    print(f"  Test:  {len(ds['test']):,}")
    print(f"  Labels: {n_labels}")

    # ============================================================
    # Label Distribution
    # ============================================================
    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION (Train)")
    print("=" * 60)

    all_labels = []
    for ex in ds["train"]:
        all_labels.extend(ex["labels"])
    label_counts = Counter(all_labels)

    label_dist = pd.DataFrame([
        {"emotion": label_names[idx], "count": count, "pct": count / len(ds["train"]) * 100}
        for idx, count in label_counts.items()
    ]).sort_values("count", ascending=False)

    print(label_dist.to_string(index=False))

    max_count = label_dist["count"].max()
    min_count = label_dist["count"].min()
    imbalance = max_count / min_count
    print(f"\nImbalance ratio: {imbalance:.1f}x")

    # ============================================================
    # Multi-label stats
    # ============================================================
    labels_per = [len(ex["labels"]) for ex in ds["train"]]
    multilabel_dist = Counter(labels_per)
    avg_labels = sum(labels_per) / len(labels_per)

    print(f"\nMulti-label stats:")
    for n, count in sorted(multilabel_dist.items()):
        print(f"  {n} label(s): {count:,} ({count/len(ds['train'])*100:.1f}%)")
    print(f"  Avg: {avg_labels:.2f}")

    # ============================================================
    # Text length stats
    # ============================================================
    lengths = [len(ex["text"].split()) for ex in ds["train"]]
    print(f"\nText length (words):")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}")
    print(f"  Mean: {sum(lengths)/len(lengths):.1f}, Median: {sorted(lengths)[len(lengths)//2]}")

    # ============================================================
    # Save outputs
    # ============================================================
    label_dist.to_csv(output_dir / "label_distribution.csv", index=False)
    with open(output_dir / "label_names.json", "w") as f:
        json.dump(label_names, f, indent=2)

    summary = {
        "dataset": "GoEmotions",
        "splits": {
            "train": len(ds["train"]),
            "validation": len(ds["validation"]),
            "test": len(ds["test"]),
        },
        "num_labels": n_labels,
        "label_names": label_names,
        "imbalance_ratio": imbalance,
        "most_common_label": label_dist.iloc[0]["emotion"],
        "rarest_label": label_dist.iloc[-1]["emotion"],
        "multilabel_distribution": dict(multilabel_dist),
        "avg_labels_per_example": avg_labels,
        "text_length_stats": {
            "min": int(min(lengths)),
            "max": int(max(lengths)),
            "mean": float(sum(lengths) / len(lengths)),
            "median": int(sorted(lengths)[len(lengths) // 2]),
        },
    }

    with open(output_dir / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ EDA outputs saved to: {output_dir}/")

    # ============================================================
    # Plot (optional)
    # ============================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_dir = Path("results/plots")
        plots_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(14, 6))
        label_dist_sorted = label_dist.sort_values("count", ascending=True)
        ax.barh(label_dist_sorted["emotion"], label_dist_sorted["count"], color="steelblue")
        ax.set_xlabel("Số samples")
        ax.set_title("GoEmotions Label Distribution (Train)")
        ax.set_xscale("log")
        plt.tight_layout()
        plt.savefig(plots_dir / "label_distribution.png", dpi=100, bbox_inches="tight")
        print(f"📊 Plot saved: {plots_dir}/label_distribution.png")
    except ImportError:
        print("⚠️  matplotlib not available, skipped plot")


if __name__ == "__main__":
    main()
