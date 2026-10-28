"""Recording -> emotion probabilities, with a fallback to "Neutral" when the model hesitates."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .features import SAMPLE_RATE, cnn_input
from .model import DEFAULT_WEIGHTS, LABELS, load_checkpoint
from .preprocess import LEGACY, PreprocessConfig, prepare

FALLBACK = "Neutral"
DEFAULT_THRESHOLD = 0.40


@dataclass(frozen=True)
class ClassifierConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    temperature: float = 1.0  # softmax temperature (calibration)
    threshold: float = DEFAULT_THRESHOLD  # below this top probability the answer is `FALLBACK`


@dataclass
class EmotionResult:
    label: str  # decision after the confidence threshold
    top_label: str  # raw argmax of the averaged probabilities
    confidence: float  # probability of `top_label`
    probs: dict[str, float]  # averaged over the windows
    uncertain: bool  # True when `label` is the fallback because of hesitation or missing speech
    reason: str = ""  # "no speech", "low confidence" or ""
    n_windows: int = 0
    speech_seconds: float = 0.0
    window_probs: list[list[float]] = field(default_factory=list)


class EmotionClassifier:
    def __init__(self, model: torch.nn.Module, config: ClassifierConfig = ClassifierConfig(), stats: tuple | None = None):
        self.model = model.eval()
        self.config = config
        self.stats = stats  # (mean, std) of the MFCCs the model was trained with, or None

    @classmethod
    def from_weights(cls, weights: str | Path = DEFAULT_WEIGHTS):
        """Rebuild the classifier exactly as it was trained: preprocessing, MFCC statistics, calibration."""
        ckpt = load_checkpoint(weights)
        preprocess = PreprocessConfig(**ckpt.preprocess) if ckpt.preprocess else LEGACY
        config = ClassifierConfig(
            preprocess=preprocess,
            temperature=ckpt.temperature,
            threshold=DEFAULT_THRESHOLD if ckpt.threshold is None else ckpt.threshold,
        )
        return cls(ckpt.model, config, ckpt.stats)

    @torch.no_grad()
    def predict_proba(self, waveforms: list[np.ndarray]) -> np.ndarray:
        """Per-window class probabilities, shape (len(waveforms), 6)."""
        batch = torch.from_numpy(np.stack([cnn_input(w, SAMPLE_RATE, self.stats) for w in waveforms]))
        logits = self.model(batch) / self.config.temperature
        return torch.softmax(logits, dim=1).numpy()

    def classify(self, audio: np.ndarray, sr: int = SAMPLE_RATE, threshold: float | None = None) -> EmotionResult:
        """Classify a mono recording (any sample rate; int16 or float).

        `threshold` overrides `config.threshold` for this call.
        """
        threshold = self.config.threshold if threshold is None else threshold
        prepared = prepare(audio, sr, self.config.preprocess)
        if not prepared.has_speech:
            return EmotionResult(
                label=FALLBACK,
                top_label=FALLBACK,
                confidence=0.0,
                probs={name: 0.0 for name in LABELS},
                uncertain=True,
                reason="no speech",
                speech_seconds=prepared.speech_s,
            )
        window_probs = self.predict_proba(prepared.windows)
        probs = window_probs.mean(axis=0)
        top = int(np.argmax(probs))
        confidence = float(probs[top])
        low = confidence < threshold
        return EmotionResult(
            label=FALLBACK if low else LABELS[top],
            top_label=LABELS[top],
            confidence=confidence,
            probs={name: float(p) for name, p in zip(LABELS, probs)},
            uncertain=low,
            reason="low confidence" if low else "",
            n_windows=len(prepared.windows),
            speech_seconds=prepared.speech_s,
            window_probs=window_probs.tolist(),
        )
