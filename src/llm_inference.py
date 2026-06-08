"""
============================================================
LLM Inference module — Llama 3.1 8B Instruct on GoEmotions
============================================================

Provides:
    - LlamaInferenceClient: Load model locally, run inference with
      checkpoint resume and configurable batch size.
    - run_inference(): Main entry-point

Usage (module):
    python -m src.llm_inference --config configs/llama_zeroshot.yaml --n_samples 2000

Features:
    - Local inference  : no API key, no rate limits
    - Checkpointing    : saves progress every N samples to JSON
    - Resume support   : skips samples already in checkpoint
    - Output           : predictions.json + metrics.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import load_dataset  # must import before torch to avoid segfault
import torch

from .data import EMOTION_NAMES, load_goemotions
from .prompts import build_prompt, parse_response
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
# Llama Client
# ============================================================
class LlamaInferenceClient:
    """
    Local Llama inference client for GoEmotions multi-label classification.

    Loads the model once at init, then runs sample-by-sample inference
    with checkpoint/resume support. No API key or rate limits required.

    Args:
        config: Parsed YAML config dict (from load_config())
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_name: str = config["experiment"]["name"]
        model_cfg = config["model"]
        # local_path takes priority over hub name for offline use
        self.model_name: str = model_cfg.get("local_path") or model_cfg["name"]
        self.max_new_tokens: int = model_cfg.get("max_new_tokens", 128)
        self.local_files_only: bool = model_cfg.get("local_files_only", False)
        self.load_in_4bit: bool = model_cfg.get("load_in_4bit", False)
        self.device: str = model_cfg.get("device", "auto")

        inf = config["inference"]
        self.n_samples: int = inf.get("n_samples", 2000)
        self.split: str = inf.get("split", "test")
        self.seed: int = inf.get("seed", 42)
        self.save_every_n: int = inf.get("save_every_n", 50)

        self.prompt_mode: str = config["prompt"]["type"]  # "zero_shot" or "few_shot"

        paths = config["paths"]
        self.output_dir = Path(paths["output_dir"])
        self.results_dir = Path(paths["results_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._init_model()

    # ----------------------------------------------------------
    def _init_model(self) -> None:
        """Load model via HuggingFace transformers pipeline."""
        from transformers import pipeline as hf_pipeline, BitsAndBytesConfig

        # Choose dtype: float16 on CUDA, float32 on CPU
        use_cuda = torch.cuda.is_available() and self.device != "cpu"
        dtype = torch.float16 if use_cuda else torch.float32

        # In transformers 5.x, any unrecognised pipeline kwarg is stored in model_kwargs
        # and later forwarded to model.generate(), causing errors.  Never pass
        # local_files_only to pipeline(); set HF_HUB_OFFLINE=1 env var for offline mode.
        import os as _os
        if self.local_files_only:
            _os.environ["HF_HUB_OFFLINE"] = "1"

        pipeline_kwargs: Dict[str, Any] = {
            "task": "text-generation",
            "model": self.model_name,
            "dtype": dtype,
            "device_map": self.device,
        }

        if self.load_in_4bit:
            if not use_cuda:
                logger.warning("load_in_4bit=True requires CUDA; ignoring on CPU.")
            else:
                pipeline_kwargs["model_kwargs"] = {
                    "quantization_config": BitsAndBytesConfig(load_in_4bit=True)
                }

        logger.info(
            "Loading model: %s  device=%s  dtype=%s  4bit=%s  offline=%s",
            self.model_name, self.device, dtype, self.load_in_4bit, self.local_files_only,
        )
        self._pipe = hf_pipeline(**pipeline_kwargs)
        self._pipe.tokenizer.pad_token_id = self._pipe.tokenizer.eos_token_id
        logger.info("Model loaded: %s (mode=%s)", self.model_name, self.prompt_mode)

    # ----------------------------------------------------------
    def _call_llama(self, text: str) -> str:
        """
        Run a single inference call.

        Applies the chat template, generates, and returns only the
        newly generated tokens (not the prompt).
        """
        from transformers import GenerationConfig

        prompt_text = build_prompt(text, mode=self.prompt_mode)
        messages = [{"role": "user", "content": prompt_text}]
        formatted_prompt = self._pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        gen_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self._pipe.tokenizer.eos_token_id,
        )
        outputs = self._pipe(
            formatted_prompt,
            generation_config=gen_config,
            return_full_text=False,
        )
        return outputs[0]["generated_text"].strip()

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

        np.random.seed(self.seed)
        total = len(split_ds)
        n = min(self.n_samples, total)
        indices = np.random.choice(total, size=n, replace=False).tolist()
        logger.info("Sampling %d / %d examples (seed=%d)", n, total, self.seed)

        checkpoint_path = self.output_dir / "checkpoint.json"
        checkpoint = _load_checkpoint(checkpoint_path)
        completed_ids = set(checkpoint["completed_ids"])
        results: List[Dict[str, Any]] = checkpoint["results"]

        skipped = len(completed_ids)
        if skipped:
            logger.info("Resuming — %d samples already done, skipping.", skipped)

        for idx in indices:
            sample = split_ds[idx]
            sample_id = sample.get("id", str(idx))

            if sample_id in completed_ids:
                continue

            text = sample["text"]
            true_labels = [EMOTION_NAMES[j] for j in sample["labels"]]

            raw_response = ""
            predicted_labels: List[str] = ["neutral"]
            try:
                raw_response = self._call_llama(text)
                predicted_labels = parse_response(raw_response)
            except Exception as exc:
                logger.error("Skipping sample %s: %s", sample_id, exc)

            results.append({
                "id": sample_id,
                "text": text,
                "true_labels": true_labels,
                "predicted_labels": predicted_labels,
                "raw_response": raw_response,
            })
            completed_ids.add(sample_id)

            processed = len(results)
            if processed % self.save_every_n == 0:
                checkpoint["results"] = results
                checkpoint["completed_ids"] = list(completed_ids)
                _save_checkpoint(checkpoint, checkpoint_path)
                logger.info(
                    "Checkpoint saved — %d / %d done (%.1f%%)",
                    processed, n, 100.0 * processed / n,
                )

        # Final checkpoint
        checkpoint["results"] = results
        checkpoint["completed_ids"] = list(completed_ids)
        _save_checkpoint(checkpoint, checkpoint_path)

        metrics = self._compute_metrics(results)

        pred_path = self.output_dir / "predictions.json"
        metrics_path = self.output_dir / "metrics.json"
        combined_metrics_path = self.results_dir / f"{self.experiment_name}_metrics.json"

        save_json(results, pred_path)
        save_json(metrics, metrics_path)
        save_json({**{"experiment": self.experiment_name}, **metrics}, combined_metrics_path)

        logger.info(
            "Done! F1-macro=%.4f  F1-micro=%.4f  Hamming=%.4f",
            metrics["f1_macro"], metrics["f1_micro"], metrics["hamming_loss"],
        )
        logger.info("Predictions → %s", pred_path)
        logger.info("Metrics     → %s", metrics_path)

        return {"results": results, "metrics": metrics}

    # ----------------------------------------------------------
    def _compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute F1-macro, F1-micro, Hamming Loss from prediction records."""
        from sklearn.metrics import f1_score, hamming_loss

        y_true, y_pred = [], []
        for rec in results:
            y_true.append(_emotions_to_multihot(rec["true_labels"]))
            y_pred.append(_emotions_to_multihot(rec["predicted_labels"]))

        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)

        per_class = f1_score(y_true_np, y_pred_np, average=None, zero_division=0).tolist()
        per_class_dict = {
            name: round(float(f), 4) for name, f in zip(EMOTION_NAMES, per_class)
        }

        return {
            "f1_macro": round(float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)), 4),
            "f1_micro": round(float(f1_score(y_true_np, y_pred_np, average="micro", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)), 4),
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
    client = LlamaInferenceClient(config)
    return client.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Llama LLM inference on GoEmotions")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (e.g. configs/llama_zeroshot.yaml)",
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
