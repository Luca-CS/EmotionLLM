"""From a raw recording to the list of windows fed to the CNN."""

from dataclasses import dataclass

import numpy as np

from .audio import detect_speech, normalise_level, resample, sliding_windows, to_mono_float, trim_silence
from .features import SAMPLE_RATE


@dataclass(frozen=True)
class PreprocessConfig:
    normalise: str = "rms"  # "rms", "peak" or "none"
    target_dbfs: float = -26.0
    trim: bool = True  # voice activity detection: cut silences
    max_pause_s: float = 0.5
    min_speech_s: float = 0.3  # below this much detected speech, the recording is rejected
    window_s: float = 6.0
    hop_s: float = 3.0
    max_duration_s: float = 60.0


# What the original code did: no normalisation, no silence removal, the first 16 s only.
LEGACY = PreprocessConfig(
    normalise="none", trim=False, min_speech_s=0.0, window_s=16.0, hop_s=16.0, max_duration_s=16.0
)


@dataclass
class Prepared:
    windows: list[np.ndarray]
    duration_s: float  # length of the input recording (capped at `max_duration_s`)
    speech_s: float  # speech detected by the VAD
    has_speech: bool


def prepare(x: np.ndarray, sr: int, cfg: PreprocessConfig = PreprocessConfig()) -> Prepared:
    """Mono 16 kHz float32 -> normalised, silence-trimmed windows of at most `cfg.window_s`."""
    x = resample(to_mono_float(x), sr, SAMPLE_RATE)[: int(cfg.max_duration_s * SAMPLE_RATE)]
    duration = len(x) / SAMPLE_RATE
    segments = detect_speech(x, SAMPLE_RATE)
    speech_s = sum(e - s for s, e in segments) / SAMPLE_RATE

    if len(x) == 0 or speech_s < cfg.min_speech_s or (cfg.trim and not segments):
        return Prepared([], duration, speech_s, False)
    if cfg.trim:
        x = trim_silence(x, segments, SAMPLE_RATE, max_pause_s=cfg.max_pause_s)
    x = normalise_level(x, cfg.normalise, cfg.target_dbfs)
    return Prepared(sliding_windows(x, SAMPLE_RATE, cfg.window_s, cfg.hop_s), duration, speech_s, True)
