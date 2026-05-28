"""
============================================================
compare_results.py — Aggregate and compare experiment metrics
============================================================

Usage:
    python scripts/compare_results.py

What it does:
    1. Scans results/metrics/ for all *_metrics.json files
    2. Prints a comparison table: Model | Method | F1-macro | F1-micro | Hamming Loss
    3. Highlights the best F1-macro row
    4. Saves table → results/metrics/comparison_table.csv
    5. Saves bar chart → results/plots/model_comparison.png
"""

import json
import sys
from pathlib import Path

# Allow running from repo root without installing as a package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for headless envs)
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# Constants
# ============================================================
METRICS_DIR = ROOT / "results" / "metrics"
PLOTS_DIR = ROOT / "results" / "plots"
CSV_OUT = METRICS_DIR / "comparison_table.csv"
PLOT_OUT = PLOTS_DIR / "model_comparison.png"

# Friendly display names: config experiment name → (Model, Method)
DISPLAY_MAP = {
    "bert_base_baseline": ("BERT-base", "Fine-tune"),
    "roberta_base_baseline": ("RoBERTa-base", "Fine-tune"),
    "gemini_zeroshot": ("Gemini Flash", "Zero-shot"),
    "gemini_fewshot_k5": ("Gemini Flash", "Few-shot (k=5)"),
}


# ============================================================
# Helpers
# ============================================================
def load_all_metrics(metrics_dir: Path) -> list[dict]:
    """Load all *_metrics.json files from metrics_dir."""
    rows = []
    pattern = "*_metrics.json"
    files = sorted(metrics_dir.glob(pattern))

    if not files:
        print(f"⚠️  No metrics files found in {metrics_dir}")
        print("   Run at least one experiment first:")
        print("   python -m src.train --config configs/bert_base.yaml")
        return rows

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            print(f"⚠️  Skipping {fpath.name}: JSON error — {exc}")
            continue

        experiment = data.get("experiment", fpath.stem.replace("_metrics", ""))
        model_name, method = DISPLAY_MAP.get(experiment, (experiment, "—"))

        rows.append({
            "Experiment": experiment,
            "Model": model_name,
            "Method": method,
            "F1-macro": data.get("f1_macro", float("nan")),
            "F1-micro": data.get("f1_micro", float("nan")),
            "Hamming Loss": data.get("hamming_loss", float("nan")),
            "N Samples": data.get("n_samples", "—"),
        })

    return rows


def print_table(df: pd.DataFrame) -> None:
    """Pretty-print the comparison table, highlighting best F1-macro."""
    if df.empty:
        return

    best_idx = df["F1-macro"].idxmax()
    print("\n" + "=" * 72)
    print(f"{'Model':<20} {'Method':<18} {'F1-macro':>10} {'F1-micro':>10} {'Hamm.Loss':>12}")
    print("-" * 72)
    for i, row in df.iterrows():
        marker = " ◀ BEST" if i == best_idx else ""
        print(
            f"{row['Model']:<20} {row['Method']:<18} "
            f"{row['F1-macro']:>10.4f} {row['F1-micro']:>10.4f} "
            f"{row['Hamming Loss']:>12.4f}{marker}"
        )
    print("=" * 72)
    print(f"\nBest F1-macro: {df.loc[best_idx, 'Model']} / {df.loc[best_idx, 'Method']}"
          f"  →  {df.loc[best_idx, 'F1-macro']:.4f}\n")


def save_bar_chart(df: pd.DataFrame, out_path: Path) -> None:
    """Save a horizontal bar chart comparing F1-macro across models."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [f"{row.Model}\n({row.Method})" for row in df.itertuples()]
    values = df["F1-macro"].tolist()
    colors = ["#4C72B0" if v != max(values) else "#DD8452" for v in values]

    fig, ax = plt.subplots(figsize=(9, max(3, len(labels) * 1.1)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.55)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=10,
        )

    ax.set_xlabel("F1-macro", fontsize=12)
    ax.set_title("Model Comparison — F1-macro (GoEmotions test set)", fontsize=13, pad=14)
    ax.set_xlim(0, min(1.0, max(values) + 0.12))
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(x=0.46, color="gray", linestyle="--", linewidth=1, label="Paper baseline (0.46)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Bar chart saved → {out_path}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {METRICS_DIR} for metrics files…")
    rows = load_all_metrics(METRICS_DIR)

    if not rows:
        print("No results to compare yet. Run some experiments first.")
        return

    df = pd.DataFrame(rows).sort_values("F1-macro", ascending=False).reset_index(drop=True)

    # Print table
    print_table(df)

    # Save CSV
    display_df = df[["Model", "Method", "F1-macro", "F1-micro", "Hamming Loss", "N Samples"]]
    display_df.to_csv(CSV_OUT, index=False)
    print(f"Comparison table saved → {CSV_OUT}")

    # Save plot
    save_bar_chart(display_df, PLOT_OUT)


if __name__ == "__main__":
    main()
