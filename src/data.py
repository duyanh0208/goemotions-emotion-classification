"""
============================================================
Data module — GoEmotions Dataset & Preprocessing
============================================================

Provides:
    - GoEmotionsDataset: PyTorch Dataset wrapper với multi-hot encoding
    - load_goemotions(): Load và optionally subset dataset
    - compute_pos_weights(): Tính pos_weight cho BCEWithLogitsLoss
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict


# ============================================================
# Constants
# ============================================================
NUM_LABELS = 28

EMOTION_NAMES = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]


# ============================================================
# Dataset Class
# ============================================================
class GoEmotionsDataset(Dataset):
    """
    PyTorch Dataset cho GoEmotions multi-label classification.

    Original format:
        item = {"text": str, "labels": List[int], "id": str}
        labels là list các index emotion (e.g., [0, 4])

    Converted format:
        item = {
            "input_ids": Tensor (max_length,),
            "attention_mask": Tensor (max_length,),
            "labels": Tensor (28,) — multi-hot binary
        }
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        max_length: int = 128,
        num_labels: int = NUM_LABELS,
    ):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        text = item["text"]
        label_indices = item["labels"]

        # Multi-hot encoding
        labels = torch.zeros(self.num_labels, dtype=torch.float)
        labels[label_indices] = 1.0

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }


# ============================================================
# Loading Functions
# ============================================================
def load_goemotions(
    dataset_name: str = "google-research-datasets/go_emotions",
    config: str = "simplified",
    debug: bool = False,
    debug_train_size: int = 1000,
    debug_val_size: int = 500,
) -> DatasetDict:
    """
    Load GoEmotions dataset từ HuggingFace.

    Args:
        dataset_name: Tên dataset trên HF Hub
        config: 'simplified' (28 classes) hoặc 'raw'
        debug: Nếu True, subset xuống nhỏ để test pipeline
        debug_train_size, debug_val_size: Kích thước subset khi debug

    Returns:
        DatasetDict với keys: train, validation, test
    """
    ds = load_dataset(dataset_name, config)

    if debug:
        print(f"⚠️  DEBUG MODE: subset train={debug_train_size}, val={debug_val_size}")
        ds["train"] = ds["train"].select(range(min(debug_train_size, len(ds["train"]))))
        ds["validation"] = ds["validation"].select(
            range(min(debug_val_size, len(ds["validation"])))
        )

    return ds


def create_dataloaders(
    ds: DatasetDict,
    tokenizer,
    max_length: int = 128,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo train/val/test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = GoEmotionsDataset(ds["train"], tokenizer, max_length)
    val_ds = GoEmotionsDataset(ds["validation"], tokenizer, max_length)
    test_ds = GoEmotionsDataset(ds["test"], tokenizer, max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# Class Imbalance Handling
# ============================================================
def compute_pos_weights(
    hf_dataset,
    num_labels: int = NUM_LABELS,
    clip_min: float = 1.0,
    clip_max: float = 50.0,
) -> torch.Tensor:
    """
    Tính pos_weight cho mỗi class trong BCEWithLogitsLoss.

    Formula:
        pos_weight[c] = (số samples không có class c) / (số samples có class c)

    Class hiếm → pos_weight cao → loss "phạt" model nặng hơn khi miss class này.

    Args:
        hf_dataset: HuggingFace dataset (train split)
        num_labels: Số classes (28)
        clip_min, clip_max: Range để clip pos_weight (tránh quá lớn)

    Returns:
        Tensor (num_labels,) — pos_weight cho mỗi class
    """
    label_counts = np.zeros(num_labels)
    total = len(hf_dataset)

    for item in hf_dataset:
        for label_idx in item["labels"]:
            label_counts[label_idx] += 1

    # Tránh chia 0 bằng epsilon
    pos_weights = (total - label_counts) / (label_counts + 1e-6)
    pos_weights = np.clip(pos_weights, clip_min, clip_max)

    return torch.tensor(pos_weights, dtype=torch.float)
