"""The two models of the original project, to measure them as they were shipped.

`emotion_classification_model_weight_2.pth` is the CNN of `rapport_LF.ipynb` (same class as `emotionllm.model`).
`emotion_classification_model_weight_3.pth` (the file that was in the repo) is the Transformer of `emo_cla_3.ipynb`,
copied here verbatim: nothing in the original repo could load it.
"""

import torch
import torch.nn as nn


class EmotionClassificationTransformer(nn.Module):
    def __init__(self, input_size=13, num_classes=6, num_heads=4, num_layers=3, hidden_size=128):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            dim_feedforward=hidden_size,
            dropout=0.1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)  # (batch, 500 frames, 13)
        x = self.embedding(x)
        out = self.transformer(x, torch.zeros_like(x))[:, -1, :]  # last position, zero decoder input
        return self.fc(out)
