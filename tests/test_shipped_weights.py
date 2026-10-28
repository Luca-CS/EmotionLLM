"""The weight files that live in the repo must load into the architecture that the code describes."""

import numpy as np
import pytest
import torch

from emotionllm.inference import EmotionClassifier
from emotionllm.model import DEFAULT_WEIGHTS, LABELS, WEIGHTS_DIR, EmotionClassificationCNN, load_checkpoint

from .conftest import speech_like

ORIGINAL = WEIGHTS_DIR / "original"


def test_shipped_checkpoint_carries_its_recipe():
    ckpt = load_checkpoint(DEFAULT_WEIGHTS)
    assert ckpt.stats is not None and ckpt.stats[0].shape == (13,) and np.all(ckpt.stats[1] > 0)
    assert ckpt.preprocess is not None and ckpt.temperature > 0
    assert ckpt.threshold is not None and 0 <= ckpt.threshold < 1
    assert "loso_balanced_accuracy" in ckpt.meta


def test_shipped_classifier_returns_valid_probabilities():
    result = EmotionClassifier.from_weights().classify(speech_like(4.0), threshold=0.0)
    assert set(result.probs) == set(LABELS)
    assert sum(result.probs.values()) == pytest.approx(1.0, abs=1e-4)
    assert result.label == result.top_label


def test_original_cnn_weights_load_strictly():
    """`weight_2` is the CNN of the notebook; loading it must not rely on `strict=False`."""
    state = torch.load(ORIGINAL / "emotion_classification_model_weight_2.pth", map_location="cpu", weights_only=True)
    EmotionClassificationCNN().load_state_dict(state, strict=True)


def test_original_weight_3_is_not_a_cnn():
    """The file shipped as the CNN was a Transformer: the old `strict=False` load silently ignored all of it."""
    state = torch.load(ORIGINAL / "emotion_classification_model_weight_3.pth", map_location="cpu", weights_only=True)
    assert any(key.startswith("transformer.") for key in state)
    with pytest.raises(RuntimeError, match="Missing key"):
        EmotionClassificationCNN().load_state_dict(state, strict=True)
