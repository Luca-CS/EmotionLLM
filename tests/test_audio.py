import numpy as np
import pytest

from emotionllm.audio import (
    detect_speech,
    normalise_level,
    resample,
    rms_dbfs,
    sliding_windows,
    to_mono_float,
    trim_silence,
)
from emotionllm.features import SAMPLE_RATE

from .conftest import silence, speech_like


def test_to_mono_float_int16_stereo():
    data = np.array([[32767, -32768], [0, 0]], dtype=np.int16)
    out = to_mono_float(data)
    assert out.dtype == np.float32 and out.shape == (2,)
    assert abs(out[0]) < 1e-3 and out[1] == 0


def test_resample_length_and_identity():
    x = speech_like(1.0)
    assert np.array_equal(resample(x, SAMPLE_RATE), x)
    assert len(resample(x[::2], 8000, SAMPLE_RATE)) == len(x) // 2 * 2
    assert len(resample(np.zeros(44_100, dtype=np.float32), 44_100, SAMPLE_RATE)) == SAMPLE_RATE


def test_vad_finds_the_speech(clip):
    segments = detect_speech(clip)
    assert len(segments) == 1
    start, end = segments[0]
    assert abs(start / SAMPLE_RATE - 1.0) < 0.35
    assert abs(end / SAMPLE_RATE - 4.0) < 0.35


def test_vad_rejects_silence_noise_and_tone():
    assert detect_speech(np.zeros(SAMPLE_RATE, dtype=np.float32)) == []
    assert detect_speech(silence(3.0)) == []
    tone = 0.3 * np.sin(2 * np.pi * 440 * np.arange(3 * SAMPLE_RATE) / SAMPLE_RATE).astype(np.float32)
    assert detect_speech(tone) == []
    assert detect_speech(np.zeros(0, dtype=np.float32)) == []


def test_vad_bridges_short_gaps_only():
    x = np.concatenate(
        [speech_like(1.0), silence(0.15), speech_like(1.0, seed=4), silence(1.0), speech_like(1.0, seed=5)]
    )
    assert len(detect_speech(x)) == 2


def test_trim_silence_shrinks_long_pauses(clip):
    x = np.concatenate([clip, silence(3.0, seed=7), speech_like(1.0, seed=8)])
    trimmed = trim_silence(x, detect_speech(x), max_pause_s=0.5)
    assert len(trimmed) / SAMPLE_RATE < len(x) / SAMPLE_RATE - 3.0
    assert len(trimmed) / SAMPLE_RATE > 4.0  # the speech itself is kept
    assert len(trim_silence(x, [])) == 0


@pytest.mark.parametrize("gain", [0.05, 1.0])
def test_rms_normalisation_reaches_target(gain):
    x = speech_like(3.0) * gain
    y = normalise_level(x, "rms", -26.0)
    assert abs(rms_dbfs(y) - (-26.0)) < 0.5
    assert np.abs(y).max() <= 0.99 + 1e-6


def test_peak_normalisation_and_none():
    x = 0.1 * speech_like(2.0)
    assert abs(np.abs(normalise_level(x, "peak")).max() - 0.95) < 1e-4
    assert normalise_level(x, "none") is x
    with pytest.raises(ValueError):
        normalise_level(x, "loudest")


def test_normalisation_leaves_digital_silence_alone():
    z = np.zeros(1000, dtype=np.float32)
    assert np.array_equal(normalise_level(z, "rms"), z)


def test_sliding_windows():
    x = np.arange(10 * SAMPLE_RATE, dtype=np.float32)
    short = np.arange(3 * SAMPLE_RATE, dtype=np.float32)
    assert len(sliding_windows(short, win_s=6, hop_s=3)) == 1
    windows = sliding_windows(x, win_s=6, hop_s=3)
    assert [len(w) for w in windows] == [6 * SAMPLE_RATE] * len(windows)
    assert windows[0][0] == 0 and windows[-1][-1] == x[-1]  # the end of the signal is covered
    # a leftover shorter than min_tail_s is not worth a window
    assert len(sliding_windows(np.zeros(int(6.5 * SAMPLE_RATE), dtype=np.float32), win_s=6, hop_s=3)) == 1
