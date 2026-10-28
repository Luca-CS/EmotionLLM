"""The CNN of the ST7 project (architecture unchanged) and its checkpoint format."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .features import N_FRAMES, N_MFCC

# Same order as the sorted IEMOCAP codes ang, exc, fru, hap, neu, sad.
LABELS = ("Angry", "Excited", "Frustrated", "Happy", "Neutral", "Sad")
WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "emotion_cnn.pth"


class EmotionClassificationCNN(nn.Module):
    """3 x (conv 3x3 + ReLU + max-pool 2x2) then 2 dense layers, on a (1, 13, 500) MFCC image."""

    def __init__(self, n_classes: int = len(LABELS)):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 1 * 62, 128)  # 13 -> 6 -> 3 -> 1 rows, 500 -> 250 -> 125 -> 62 columns
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 1, 13, 500) -> logits (batch, n_classes)."""
        if x.shape[1:] != (1, N_MFCC, N_FRAMES):
            raise ValueError(f"expected input of shape (batch, 1, {N_MFCC}, {N_FRAMES}), got {tuple(x.shape)}")
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.relu(self.fc1(x.flatten(1)))
        return self.fc2(self.dropout2(x))


@dataclass
class Checkpoint:
    """Weights plus everything needed to reproduce the inputs the network was trained on."""

    model: EmotionClassificationCNN
    stats: tuple[np.ndarray, np.ndarray] | None = None  # MFCC mean/std per coefficient
    preprocess: dict | None = None  # PreprocessConfig fields; None = the original untouched pipeline
    temperature: float = 1.0
    threshold: float | None = None  # confidence below which the answer falls back to Neutral
    meta: dict = field(default_factory=dict)


def save_checkpoint(path: str | Path, ckpt: Checkpoint) -> None:
    payload = {
        "state_dict": ckpt.model.state_dict(),
        "preprocess": ckpt.preprocess,
        "temperature": float(ckpt.temperature),
        "threshold": None if ckpt.threshold is None else float(ckpt.threshold),
        "meta": ckpt.meta,
    }
    if ckpt.stats is not None:
        payload["mean"] = torch.as_tensor(np.asarray(ckpt.stats[0], dtype=np.float32))
        payload["std"] = torch.as_tensor(np.asarray(ckpt.stats[1], dtype=np.float32))
    torch.save(payload, path)


def load_checkpoint(path: str | Path = DEFAULT_WEIGHTS) -> Checkpoint:
    """Read a checkpoint written by `save_checkpoint`, or a bare state dict of the original project."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    model = EmotionClassificationCNN()
    if "state_dict" not in obj:
        model.load_state_dict(obj)
        return Checkpoint(model.eval())
    model.load_state_dict(obj["state_dict"])
    stats = (obj["mean"].numpy(), obj["std"].numpy()) if "mean" in obj else None
    return Checkpoint(
        model.eval(), stats, obj.get("preprocess"), obj.get("temperature", 1.0), obj.get("threshold"), obj.get("meta", {})
    )


def load_model(weights: str | Path = DEFAULT_WEIGHTS) -> EmotionClassificationCNN:
    return load_checkpoint(weights).model
