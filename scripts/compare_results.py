"""
============================================================
compare_results.py — Aggregate and compare experiment metrics
============================================================

Usage:
    python scripts/compare_results.py

What it does:
    1. Scans results/metrics/ for all *_metrics.json files
    2. Loads threshold-tuning results if available
    3. Prints comparison table: Model | Method | Threshold | F1-macro | F1-micro | Hamming Loss
    4. Highlights the best F1-macro row
    5. Saves table → results/metrics/comparison_table.csv
    6. Saves bar chart → results/plots/model_comparison.png
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# Constants
# ============================================================
METRICS_DIR = ROOT / "results" / "metrics"
PLOTS_DIR = ROOT / "results" / "plots"
CSV_OUT = METRICS_DIR / "comparison_table.csv"
PLOT_OUT = PLOTS_DIR / "model_comparison.png"

DISPLAY_MAP = {
    "bert_base_baseline": ("BERT-base", "Fine-tune", "t=0.5"),
    "roberta_base_baseline": ("RoBERTa-base", "Fine-tune", "t=0.5"),
    "gemini_zeroshot": ("Gemini 2.0 Flash", "Zero-shot (API)", "—"),
    "llama_zeroshot": ("Llama 3.2 3B", "Zero-shot", "—"),
    "qwen_zeroshot": ("Qwen2.5 3B", "Zero-shot", "—"),
    "llama_fewshot": ("Llama 3.2 3B", "Few-shot (k=5)", "—"),
    "qwen_fewshot": ("Qwen2.5 3B", "Few-shot (k=5)", "—"),
}

# Threshold-tuning result files: stem → (Model, Method, threshold label)
THRESHOLD_MAP = {
    "bert_base_threshold_tuning": ("BERT-base", "Fine-tune", "t=0.9†"),
    "roberta_base_threshold_tuning": ("RoBERTa-base", "Fine-tune", "t=0.9†"),
}


# ============================================================
# Helpers
# ============================================================
def _extract_metrics(data: dict) -> dict:
    """
    Extract flat metrics dict from either format:
    - LLM format: {"f1_macro": ..., "f1_micro": ..., "n_samples": ...}
    - Train format: {"test_metrics": {"f1_macro": ..., ...}, "test_per_class_f1": {...}}
    """
    if "f1_macro" in data:
        return data
    if "test_metrics" in data:
        m = data["test_metrics"]
        n = data.get("config", {}).get("data", {})
        return {
            "f1_macro": m.get("f1_macro", float("nan")),
            "f1_micro": m.get("f1_micro", float("nan")),
            "f1_weighted": m.get("f1_weighted", float("nan")),
            "hamming_loss": m.get("hamming_loss", float("nan")),
            "n_samples": "full test",
            "per_class_f1": data.get("test_per_class_f1", {}),
        }
    return {}


def load_all_metrics(metrics_dir: Path) -> list[dict]:
    """Load baseline metrics + threshold-tuning results."""
    rows = []
    seen: set[str] = set()

    files = sorted(metrics_dir.glob("*.json"))
    if not files:
        print(f"⚠️  No metrics files found in {metrics_dir}")
        return rows

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            print(f"⚠️  Skipping {fpath.name}: JSON error — {exc}")
            continue

        stem = fpath.stem

        # ── Threshold-tuning files ──────────────────────────────────
        if stem in THRESHOLD_MAP:
            model_name, method, thresh_label = THRESHOLD_MAP[stem]
            g = data.get("global_tuned", {})
            if g and stem not in seen:
                seen.add(stem)
                rows.append({
                    "Experiment": stem,
                    "Model": model_name,
                    "Method": method,
                    "Threshold": thresh_label,
                    "F1-macro": g.get("f1_macro", float("nan")),
                    "F1-micro": g.get("f1_micro", float("nan")),
                    "Hamming Loss": float("nan"),
                    "N Samples": data.get("n_test_samples", "—"),
                })
            continue

        # ── Baseline experiment files ───────────────────────────────
        experiment = data.get("experiment", stem.replace("_metrics", ""))
        if experiment not in DISPLAY_MAP or experiment in seen:
            continue

        m = _extract_metrics(data)
        if not m:
            continue

        n_samples = m.get("n_samples", "—")
        if isinstance(n_samples, int) and n_samples < 100:
            print(f"  Skipping {fpath.name} (n_samples={n_samples}, debug run)")
            continue

        seen.add(experiment)
        model_name, method, thresh_label = DISPLAY_MAP[experiment]
        rows.append({
            "Experiment": experiment,
            "Model": model_name,
            "Method": method,
            "Threshold": thresh_label,
            "F1-macro": m.get("f1_macro", float("nan")),
            "F1-micro": m.get("f1_micro", float("nan")),
            "Hamming Loss": m.get("hamming_loss", float("nan")),
            "N Samples": n_samples,
        })

    return rows


def print_table(df: pd.DataFrame) -> None:
    """Pretty-print the comparison table, highlighting best F1-macro."""
    if df.empty:
        return

    best_idx = df["F1-macro"].idxmax()
    print("\n" + "=" * 82)
    print(f"{'Model':<20} {'Method':<18} {'Thresh':>8} {'F1-macro':>10} {'F1-micro':>10} {'Hamm.Loss':>12}")
    print("-" * 82)
    for i, row in df.iterrows():
        marker = " ◀ BEST" if i == best_idx else ""
        hl = row.get("Hamming Loss", float("nan"))
        hl_str = f"{hl:>12.4f}" if hl == hl else f"{'—':>12}"
        print(
            f"{row['Model']:<20} {row['Method']:<18} "
            f"{row.get('Threshold', '—'):>8} "
            f"{row['F1-macro']:>10.4f} {row['F1-micro']:>10.4f} "
            f"{hl_str}{marker}"
        )
    print("=" * 82)
    print(f"† threshold tuned on test set (upper bound)")
    print(f"\nBest F1-macro: {df.loc[best_idx, 'Model']} / {df.loc[best_idx, 'Method']}"
          f" [{df.loc[best_idx, 'Threshold']}]"
          f"  →  {df.loc[best_idx, 'F1-macro']:.4f}\n")


def save_bar_chart(df: pd.DataFrame, out_path: Path) -> None:
    """Save a horizontal bar chart comparing F1-macro across models."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [f"{row.Model}\n({row.Method}, {row.Threshold})" for row in df.itertuples()]
    values = df["F1-macro"].tolist()
    max_val = max(values)

    # Color: orange = best, light blue = t=0.9 tuned, dark blue = others
    colors = []
    for row, v in zip(df.itertuples(), values):
        if v == max_val:
            colors.append("#DD8452")       # best
        elif "0.9" in str(getattr(row, "Threshold", "")):
            colors.append("#6EB5D9")       # threshold-tuned
        else:
            colors.append("#4C72B0")       # baseline

    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 1.1)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.55)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left", fontsize=10,
        )

    ax.set_xlabel("F1-macro", fontsize=12)
    ax.set_title("Model Comparison — F1-macro (GoEmotions test set)", fontsize=13, pad=14)
    ax.set_xlim(0, min(1.0, max_val + 0.14))
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(x=0.46, color="gray", linestyle="--", linewidth=1, label="Paper baseline (0.46)")
    ax.axvline(x=0.5167, color="orange", linestyle=":", linewidth=1, label="BERT t=0.9 (0.5167)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#DD8452", label="Best result"),
        Patch(facecolor="#6EB5D9", label="Threshold-tuned (t=0.9†)"),
        Patch(facecolor="#4C72B0", label="Baseline (t=0.5)"),
        plt.Line2D([0], [0], color="gray", linestyle="--", label="Paper baseline (0.46)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

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
    display_df = df[["Model", "Method", "Threshold", "F1-macro", "F1-micro", "Hamming Loss", "N Samples"]]
    display_df.to_csv(CSV_OUT, index=False)
    print(f"Comparison table saved → {CSV_OUT}")

    # Save plot
    save_bar_chart(display_df, PLOT_OUT)


if __name__ == "__main__":
    main()
