import numpy as np
import pytest
import torch

from emotionllm.features import N_FRAMES, N_MFCC, SAMPLE_RATE
from emotionllm.inference import FALLBACK, ClassifierConfig, EmotionClassifier
from emotionllm.model import LABELS, EmotionClassificationCNN
from emotionllm.preprocess import LEGACY, PreprocessConfig, prepare

from .conftest import silence, speech_like


class FixedModel(torch.nn.Module):
    """Returns the same logits for every window."""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, x):
        assert x.shape[1:] == (1, N_MFCC, N_FRAMES)
        return self.logits.repeat(x.shape[0], 1)


def classifier(logits, **cfg):
    return EmotionClassifier(FixedModel(logits), ClassifierConfig(**cfg))


def test_prepare_cuts_silence_and_windows(clip):
    p = prepare(clip, SAMPLE_RATE)
    assert p.has_speech and p.speech_s > 2.0
    assert sum(len(w) for w in p.windows) / SAMPLE_RATE < 5.0  # shorter than the raw 5 s clip


def test_prepare_long_input_gives_overlapping_windows():
    x = np.concatenate([speech_like(14.0, seed=2), silence(0.1)])
    p = prepare(x, SAMPLE_RATE, PreprocessConfig(window_s=6.0, hop_s=3.0))
    assert len(p.windows) >= 3 and all(len(w) <= 6 * SAMPLE_RATE for w in p.windows)


def test_prepare_resamples_and_accepts_int16():
    x16 = (speech_like(2.0) * 32767).astype(np.int16)
    p = prepare(x16, SAMPLE_RATE)
    p8 = prepare(x16[::2], 8000)
    assert p.has_speech and p8.has_speech
    assert abs(sum(map(len, p.windows)) - sum(map(len, p8.windows))) < 0.3 * SAMPLE_RATE


def test_prepare_no_speech_and_empty():
    assert not prepare(silence(3.0), SAMPLE_RATE).has_speech
    assert not prepare(np.zeros(0, dtype=np.float32), SAMPLE_RATE).has_speech


def test_legacy_config_keeps_the_signal_untouched():
    x = speech_like(20.0)
    p = prepare(x, SAMPLE_RATE, LEGACY)
    assert len(p.windows) == 1 and len(p.windows[0]) == 16 * SAMPLE_RATE
    assert np.array_equal(p.windows[0], x[: 16 * SAMPLE_RATE])


def test_confident_prediction(clip):
    result = classifier([0, 0, 0, 0, 0, 6.0]).classify(clip)
    assert result.label == "Sad" and not result.uncertain
    assert result.confidence > 0.9
    assert sum(result.probs.values()) == pytest.approx(1.0, abs=1e-5)
    assert set(result.probs) == set(LABELS)


def test_hesitation_falls_back_to_neutral(clip):
    result = classifier([1.0, 0.9, 0.8, 0.7, 0.6, 1.1]).classify(clip)
    assert result.uncertain and result.label == FALLBACK
    assert result.top_label == "Sad" and result.reason == "low confidence"


def test_threshold_override(clip):
    clf = classifier([1.0, 0.9, 0.8, 0.7, 0.6, 1.1])
    assert clf.classify(clip, threshold=0.0).label == "Sad"
    assert clf.classify(clip, threshold=0.9).label == FALLBACK


def test_temperature_flattens(clip):
    sharp = classifier([0, 0, 0, 0, 0, 3.0], temperature=1.0).classify(clip).confidence
    flat = classifier([0, 0, 0, 0, 0, 3.0], temperature=5.0).classify(clip).confidence
    assert flat < sharp


def test_no_speech_returns_fallback_without_calling_the_model():
    class Boom(torch.nn.Module):
        def forward(self, x):
            raise AssertionError("the model must not run on silence")

    result = EmotionClassifier(Boom()).classify(silence(2.0))
    assert result.label == FALLBACK and result.reason == "no speech" and result.n_windows == 0


def test_windows_are_averaged():
    x = speech_like(15.0, seed=9)
    result = classifier([0, 0, 4.0, 0, 0, 0]).classify(x)
    assert result.n_windows > 1 and len(result.window_probs) == result.n_windows
    assert result.label == "Frustrated"


def test_real_architecture_forward_shape_and_validation():
    model = EmotionClassificationCNN().eval()
    assert model(torch.zeros(2, 1, N_MFCC, N_FRAMES)).shape == (2, len(LABELS))
    with pytest.raises(ValueError):
        model(torch.zeros(2, 1, N_MFCC, 400))
