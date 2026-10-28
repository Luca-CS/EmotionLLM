import numpy as np
import pytest

from emotionllm.features import SAMPLE_RATE


def speech_like(seconds: float, seed: int = 0, level: float = 0.3) -> np.ndarray:
    """Syllable-rate amplitude-modulated harmonics + a little noise: enough level contrast for the VAD."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    carrier = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((140, 280, 420, 560), 1))
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 3.5 * t))
    x = level * envelope * carrier / 2 + 0.002 * rng.standard_normal(len(t))
    return x.astype(np.float32)


def silence(seconds: float, seed: int = 1, level: float = 0.002) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (level * rng.standard_normal(int(seconds * SAMPLE_RATE))).astype(np.float32)


@pytest.fixture
def clip():
    """1 s silence + 3 s speech-like + 1 s silence."""
    return np.concatenate([silence(1.0), speech_like(3.0), silence(1.0, seed=2)])
