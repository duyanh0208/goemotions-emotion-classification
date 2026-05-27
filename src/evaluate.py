"""
============================================================
Evaluation module — Metrics cho multi-label classification
============================================================

Provides:
    - compute_metrics(): F1-macro/micro/weighted, Hamming Loss
    - per_class_f1(): F1 cho từng class
    - confusion_matrix_per_class(): Binary confusion matrix mỗi class
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    classification_report,
    multilabel_confusion_matrix,
)

from .data import EMOTION_NAMES, NUM_LABELS


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Tính các metrics chính cho multi-label classification.

    Args:
        logits: (N, num_labels) — raw logits
        labels: (N, num_labels) — multi-hot ground truth (0/1)
        threshold: Cutoff sigmoid → binary prediction

    Returns:
        Dict với F1-macro, F1-micro, F1-weighted, Hamming Loss
    """
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    true = labels.cpu().numpy().astype(int)

    return {
        "f1_macro": f1_score(true, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(true, preds, average="micro", zero_division=0),
        "f1_weighted": f1_score(true, preds, average="weighted", zero_division=0),
        "hamming_loss": hamming_loss(true, preds),
    }


def per_class_f1(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    class_names: List[str] = None,
) -> Dict[str, float]:
    """
    F1 cho từng class — để identify class nào model làm tốt/tệ.

    Returns:
        Dict mapping class_name → F1 score
    """
    if class_names is None:
        class_names = EMOTION_NAMES

    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    true = labels.cpu().numpy().astype(int)

    f1_per_class = f1_score(true, preds, average=None, zero_division=0)
    return {name: float(f1) for name, f1 in zip(class_names, f1_per_class)}


def confusion_matrix_per_class(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    class_names: List[str] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Binary confusion matrix cho mỗi class.

    Returns:
        Dict mapping class_name → {tn, fp, fn, tp}
    """
    if class_names is None:
        class_names = EMOTION_NAMES

    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    true = labels.cpu().numpy().astype(int)

    cms = multilabel_confusion_matrix(true, preds)  # (num_classes, 2, 2)

    result = {}
    for i, name in enumerate(class_names):
        cm = cms[i]
        result[name] = {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }
    return result


def get_full_classification_report(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
    class_names: List[str] = None,
) -> str:
    """Format string report theo style sklearn."""
    if class_names is None:
        class_names = EMOTION_NAMES

    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    true = labels.cpu().numpy().astype(int)

    return classification_report(
        true, preds,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    """
    Evaluate model trên 1 dataloader.

    Returns:
        metrics: Dict các metrics + loss
        all_logits: Tensor (N, num_labels)
        all_labels: Tensor (N, num_labels)
    """
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / len(dataloader)

    return metrics, all_logits, all_labels
