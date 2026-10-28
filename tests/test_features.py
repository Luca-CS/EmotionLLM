import numpy as np
import pytest

from emotionllm.features import HOP_LENGTH, N_FRAMES, N_MFCC, SAMPLE_RATE, cnn_input, mfcc, pad_or_crop

from .conftest import speech_like


def test_shape_and_frame_count():
    y = speech_like(2.0)
    m = mfcc(y)
    assert m.shape == (N_MFCC, 1 + len(y) // HOP_LENGTH)
    assert m.dtype == np.float32


def test_matches_librosa():
    librosa = pytest.importorskip("librosa")
    y = speech_like(5.0, seed=3)
    ref = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    assert np.abs(mfcc(y) - ref).max() < 1e-3


def test_matches_librosa_on_noise_and_loud_signal():
    librosa = pytest.importorskip("librosa")
    rng = np.random.default_rng(5)
    for y in (0.5 * rng.standard_normal(SAMPLE_RATE * 2).astype(np.float32), 0.01 * speech_like(2.0)):
        ref = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
        assert np.abs(mfcc(y) - ref).max() < 1e-2


def test_wrong_sample_rate_is_rejected():
    with pytest.raises(ValueError, match="16000"):
        mfcc(speech_like(1.0), sr=44_100)


def test_pad_and_crop():
    short = np.ones((N_MFCC, 100), dtype=np.float32)
    padded = pad_or_crop(short)
    assert padded.shape == (N_MFCC, N_FRAMES)
    assert padded[:, :100].min() == 1 and padded[:, 100:].max() == 0  # zero padding on the right
    assert pad_or_crop(np.ones((N_MFCC, 900), dtype=np.float32)).shape == (N_MFCC, N_FRAMES)


def test_cnn_input_shape():
    assert cnn_input(speech_like(3.0)).shape == (1, N_MFCC, N_FRAMES)
