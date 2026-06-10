"""
============================================================
run_multiseed.py — Multi-seed training for statistical rigor
============================================================

Runs BERT/RoBERTa training 3 times with different seeds,
then reports mean ± std across seeds.

Usage:
    python scripts/run_multiseed.py --config configs/bert_base.yaml
    python scripts/run_multiseed.py --config configs/roberta_base.yaml --seeds 42 123 456
    python scripts/run_multiseed.py --config configs/bert_base.yaml --dry-run

Why: A single-seed result may be noise. Mean ± std over ≥3 seeds gives
statistically meaningful comparison (e.g., BERT vs RoBERTa).
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, save_json

DEFAULT_SEEDS = [42, 123, 456]


def run_one_seed(config_path: str, seed: int, dry_run: bool = False) -> dict:
    """Run a single training with the given seed, return test metrics."""
    config = load_config(config_path)
    exp_name = config["experiment"]["name"]
    output_dir = Path(config["paths"]["output_dir"]) / f"seed_{seed}"
    results_dir = Path(config["paths"]["results_dir"])

    if dry_run:
        print(f"  [DRY RUN] Would run: seed={seed}, output_dir={output_dir}")
        # Return fake metrics for testing
        return {"f1_macro": 0.42 + seed/10000, "f1_micro": 0.46, "f1_weighted": 0.53, "hamming_loss": 0.078}

    cmd = [
        sys.executable, "-m", "src.train",
        "--config", config_path,
        "--no-wandb",
    ]

    # Build env override via temp config
    temp_config = copy.deepcopy(config)
    temp_config["training"]["seed"] = seed
    temp_config["paths"]["output_dir"] = str(output_dir)
    temp_config["logging"]["use_wandb"] = False

    temp_cfg_path = ROOT / f".tmp_seed_{seed}.yaml"
    import yaml
    with open(temp_cfg_path, "w") as f:
        yaml.dump(temp_config, f)

    cmd = [sys.executable, "-m", "src.train", "--config", str(temp_cfg_path), "--no-wandb"]
    print(f"\n  Running seed={seed}  (output → {output_dir})")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)

    temp_cfg_path.unlink(missing_ok=True)

    # Load results
    results_path = results_dir / f"{exp_name}.json"
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        return data.get("test_metrics", {})
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_name = config["experiment"]["name"]

    print(f"\n{'='*60}")
    print(f"Multi-seed training: {exp_name}")
    print(f"Seeds: {args.seeds}")
    print(f"{'='*60}")

    all_metrics = []
    for seed in args.seeds:
        metrics = run_one_seed(args.config, seed, args.dry_run)
        if metrics:
            all_metrics.append(metrics)
            print(f"  Seed {seed}: F1-macro={metrics.get('f1_macro', 'N/A'):.4f}")

    if not all_metrics:
        print("No results collected.")
        return

    metrics_keys = ["f1_macro", "f1_micro", "f1_weighted", "hamming_loss"]
    print(f"\n{'='*60}")
    print(f"  RESULTS ACROSS {len(all_metrics)} SEEDS")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*54}")

    summary = {}
    for key in metrics_keys:
        vals = [m[key] for m in all_metrics if key in m]
        if not vals:
            continue
        arr = np.array(vals)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "per_seed": {str(s): float(v) for s, v in zip(args.seeds, vals)},
        }
        print(f"  {key:<20} {arr.mean():>8.4f} {arr.std():>8.4f} {arr.min():>8.4f} {arr.max():>8.4f}")

    out_path = ROOT / "results" / "metrics" / f"{exp_name}_multiseed.json"
    save_json({
        "experiment": exp_name,
        "seeds": args.seeds,
        "n_runs": len(all_metrics),
        "summary": summary,
        "per_seed_raw": all_metrics,
    }, out_path)
    print(f"\n  Saved → {out_path}")
    print(f"\n  Report format: F1-macro = {summary['f1_macro']['mean']:.4f} ± {summary['f1_macro']['std']:.4f}")


if __name__ == "__main__":
    main()
