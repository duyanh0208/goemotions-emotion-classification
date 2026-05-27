"""
============================================================
Models module — Emotion Classifier Architectures
============================================================

Provides:
    - EmotionClassifier: BERT/RoBERTa + classification head
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class EmotionClassifier(nn.Module):
    """
    Generic emotion classifier với pre-trained encoder + linear head.

    Architecture:
        text → encoder (BERT/RoBERTa/...) → [CLS] embedding
             → dropout → linear → logits (num_labels,)

    Output: raw logits (chưa apply sigmoid).
    BCEWithLogitsLoss sẽ tự apply sigmoid (numerically stable).
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 28,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        hidden_size = self.encoder.config.hidden_size  # 768 (base) hoặc 1024 (large)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Init classifier weights (encoder đã có pretrained weights)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            logits: (batch, num_labels)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Lấy [CLS] token embedding (vị trí 0)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        x = self.dropout(cls_embedding)
        logits = self.classifier(x)  # (batch, num_labels)
        return logits

    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
