"""
============================================================
LLM Inference module — Gemini 2.0 Flash on GoEmotions
============================================================

Provides:
    - GeminiInferenceClient: Load config, call Gemini API with
      rate-limiting, retry logic, and checkpoint resume.
    - run_inference(): Main entry-point

Usage (module):
    python -m src.llm_inference --config configs/gemini_zeroshot.yaml --n_samples 2000

Features:
    - Rate limiting  : 15 RPM (free-tier) via time.sleep()
    - Retry logic    : 3 attempts with exponential backoff
    - Checkpointing  : saves progress every N samples to JSON
    - Resume support : skips samples already in checkpoint
    - Output         : predictions.json + metrics.json
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Load .env before any google-generativeai import
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types
from datasets import load_dataset

from .data import EMOTION_NAMES, NUM_LABELS, load_goemotions
from .prompts import EMOTION_LIST, build_prompt
from .utils import load_config, save_json

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


# ============================================================
# Helpers
# ============================================================
def _parse_gemini_response(raw: str) -> List[str]:
    """
    Parse Gemini JSON response → list of emotion strings.

    Expected format: {"emotions": ["joy", "excitement"]}
    Falls back gracefully if parsing fails.
    """
    try:
        data = json.loads(raw)
        emotions = data.get("emotions", [])
        # Validate: keep only known emotion labels
        valid = [e for e in emotions if e in EMOTION_LIST]
        return valid if valid else ["neutral"]
    except (json.JSONDecodeError, AttributeError, TypeError):
        logger.warning("Failed to parse Gemini response: %r", raw[:200])
        return ["neutral"]


def _emotions_to_multihot(emotions: List[str], emotion_names: List[str] = None) -> List[int]:
    """Convert list of emotion strings → multi-hot binary vector."""
    if emotion_names is None:
        emotion_names = EMOTION_NAMES
    vec = [0] * len(emotion_names)
    name_to_idx = {name: i for i, name in enumerate(emotion_names)}
    for e in emotions:
        if e in name_to_idx:
            vec[name_to_idx[e]] = 1
    return vec


def _load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    """Load existing checkpoint or return empty structure."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"results": [], "completed_ids": []}


def _save_checkpoint(checkpoint: Dict[str, Any], checkpoint_path: Path) -> None:
    """Atomic save to avoid corruption: write to .tmp then rename."""
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    tmp_path.replace(checkpoint_path)


# ============================================================
# Gemini Client
# ============================================================
class GeminiInferenceClient:
    """
    Gemini API client for GoEmotions multi-label classification.

    Args:
        config: Parsed YAML config dict (from load_config())
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_name: str = config["experiment"]["name"]
        self.model_name: str = config["model"]["name"]
        self.temperature: float = config["model"].get("temperature", 0.0)
        self.max_output_tokens: int = config["model"].get("max_output_tokens", 256)
        self.response_mime_type: str = config["model"].get(
            "response_mime_type", "application/json"
        )

        inf = config["inference"]
        self.n_samples: int = inf.get("n_samples", 2000)
        self.split: str = inf.get("split", "test")
        self.seed: int = inf.get("seed", 42)
        self.retry_max_attempts: int = inf.get("retry_max_attempts", 3)
        self.retry_delay_seconds: float = inf.get("retry_delay_seconds", 5.0)
        self.rate_limit_rpm: int = inf.get("rate_limit_rpm", 15)
        self.save_every_n: int = inf.get("save_every_n", 50)

        self.prompt_mode: str = config["prompt"]["type"]  # "zero_shot" or "few_shot"

        paths = config["paths"]
        self.output_dir = Path(paths["output_dir"])
        self.results_dir = Path(paths["results_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Seconds between calls to respect rate limit
        self._min_interval: float = 60.0 / self.rate_limit_rpm  # e.g. 4.0 s for 15 RPM
        self._last_call_time: float = 0.0

        self._init_gemini()

    # ----------------------------------------------------------
    def _init_gemini(self) -> None:
        """Configure the Gemini SDK (google-genai). Reads GEMINI_API_KEY from env."""
        import os

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not found. Set it in your .env file or environment."
            )
        self._client = genai.Client(api_key=api_key)
        self._gen_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type=self.response_mime_type,
        )
        logger.info("Gemini model loaded: %s (mode=%s)", self.model_name, self.prompt_mode)

    # ----------------------------------------------------------
    def _rate_limit_sleep(self) -> None:
        """Sleep to enforce the per-minute rate limit."""
        elapsed = time.time() - self._last_call_time
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)

    # ----------------------------------------------------------
    def _call_gemini(self, prompt: str) -> str:
        """
        Call Gemini API with exponential-backoff retry logic.

        Returns:
            Raw response text string.

        Raises:
            RuntimeError if all retries are exhausted.
        """
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                self._rate_limit_sleep()
                self._last_call_time = time.time()
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self._gen_config,
                )
                return response.text
            except Exception as exc:
                wait_time = self.retry_delay_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Gemini API error (attempt %d/%d): %s — retrying in %.0fs",
                    attempt,
                    self.retry_max_attempts,
                    str(exc),
                    wait_time,
                )
                if attempt < self.retry_max_attempts:
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"Gemini API failed after {self.retry_max_attempts} attempts: {exc}"
                    ) from exc
        return ""  # unreachable

    # ----------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """
        Run full inference over the dataset split.

        Returns:
            Dict with 'results' and 'metrics'.
        """
        logger.info("Loading GoEmotions dataset (split=%s)…", self.split)
        ds_full = load_goemotions()
        split_ds = ds_full[self.split]

        # Subsample reproducibly
        np.random.seed(self.seed)
        total = len(split_ds)
        n = min(self.n_samples, total)
        indices = np.random.choice(total, size=n, replace=False).tolist()
        logger.info("Sampling %d / %d examples (seed=%d)", n, total, self.seed)

        # Resume from checkpoint
        checkpoint_path = self.output_dir / "checkpoint.json"
        checkpoint = _load_checkpoint(checkpoint_path)
        completed_ids = set(checkpoint["completed_ids"])
        results: List[Dict[str, Any]] = checkpoint["results"]

        skipped = len(completed_ids)
        if skipped:
            logger.info("Resuming — %d samples already done, skipping.", skipped)

        # Inference loop
        for i, idx in enumerate(indices):
            sample = split_ds[idx]
            sample_id = sample.get("id", str(idx))

            if sample_id in completed_ids:
                continue

            text = sample["text"]
            true_label_indices = sample["labels"]
            true_labels = [EMOTION_NAMES[j] for j in true_label_indices]

            prompt = build_prompt(text, mode=self.prompt_mode)

            raw_response = ""
            predicted_labels: List[str] = ["neutral"]
            try:
                raw_response = self._call_gemini(prompt)
                predicted_labels = _parse_gemini_response(raw_response)
            except RuntimeError as exc:
                logger.error("Skipping sample %s due to API error: %s", sample_id, exc)

            record: Dict[str, Any] = {
                "id": sample_id,
                "text": text,
                "true_labels": true_labels,
                "predicted_labels": predicted_labels,
                "raw_response": raw_response,
            }
            results.append(record)
            completed_ids.add(sample_id)

            # Checkpoint
            processed = len(results)
            if processed % self.save_every_n == 0:
                checkpoint["results"] = results
                checkpoint["completed_ids"] = list(completed_ids)
                _save_checkpoint(checkpoint, checkpoint_path)
                logger.info(
                    "Checkpoint saved — %d / %d done (%.1f%%)",
                    processed,
                    n,
                    100.0 * processed / n,
                )

        # Final checkpoint
        checkpoint["results"] = results
        checkpoint["completed_ids"] = list(completed_ids)
        _save_checkpoint(checkpoint, checkpoint_path)

        # Compute metrics
        metrics = self._compute_metrics(results)

        # Save outputs
        pred_path = self.output_dir / "predictions.json"
        metrics_path = self.output_dir / "metrics.json"
        combined_metrics_path = self.results_dir / f"{self.experiment_name}_metrics.json"

        save_json(results, pred_path)
        save_json(metrics, metrics_path)
        save_json({**{"experiment": self.experiment_name}, **metrics}, combined_metrics_path)

        logger.info(
            "Done! F1-macro=%.4f  F1-micro=%.4f  Hamming=%.4f",
            metrics["f1_macro"],
            metrics["f1_micro"],
            metrics["hamming_loss"],
        )
        logger.info("Predictions → %s", pred_path)
        logger.info("Metrics     → %s", metrics_path)

        return {"results": results, "metrics": metrics}

    # ----------------------------------------------------------
    def _compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute F1-macro, F1-micro, Hamming Loss from prediction records."""
        from sklearn.metrics import f1_score, hamming_loss

        y_true, y_pred = [], []
        for rec in results:
            y_true.append(_emotions_to_multihot(rec["true_labels"]))
            y_pred.append(_emotions_to_multihot(rec["predicted_labels"]))

        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)

        # Per-class F1 (for detailed breakdown)
        per_class = f1_score(y_true_np, y_pred_np, average=None, zero_division=0).tolist()
        per_class_dict = {
            name: round(float(f), 4) for name, f in zip(EMOTION_NAMES, per_class)
        }

        return {
            "f1_macro": round(
                float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)), 4
            ),
            "f1_micro": round(
                float(f1_score(y_true_np, y_pred_np, average="micro", zero_division=0)), 4
            ),
            "f1_weighted": round(
                float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)), 4
            ),
            "hamming_loss": round(float(hamming_loss(y_true_np, y_pred_np)), 4),
            "n_samples": len(results),
            "per_class_f1": per_class_dict,
        }


# ============================================================
# CLI entry-point
# ============================================================
def run_inference(
    config_path: str,
    n_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Public API for running inference from Python code.

    Args:
        config_path: Path to YAML config file
        n_samples: Override n_samples from config (optional)

    Returns:
        Dict with 'results' and 'metrics'.
    """
    config = load_config(config_path)
    if n_samples is not None:
        config["inference"]["n_samples"] = n_samples
    client = GeminiInferenceClient(config)
    return client.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini LLM inference on GoEmotions")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (e.g. configs/gemini_zeroshot.yaml)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Override n_samples from config",
    )
    args = parser.parse_args()
    run_inference(args.config, args.n_samples)


if __name__ == "__main__":
    main()
