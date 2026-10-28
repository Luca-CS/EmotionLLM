"""Audio primitives: decoding, resampling, level normalisation, voice activity detection, windows."""

from math import gcd
from pathlib import Path

import numpy as np

from .features import SAMPLE_RATE

EPS = 1e-9


def to_mono_float(data: np.ndarray) -> np.ndarray:
    """int16/int32/float samples, (n,) or (n, channels) -> float32 mono in [-1, 1]."""
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max + 1)
    else:
        data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data)


def resample(x: np.ndarray, orig_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if orig_sr == target_sr:
        return x.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    g = gcd(int(orig_sr), int(target_sr))
    return resample_poly(x, target_sr // g, orig_sr // g).astype(np.float32)


def load_audio(path: str | Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Read any audio file as 16 kHz mono float32 (libsndfile first, PyAV as a fallback)."""
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return resample(to_mono_float(data), sr, target_sr)
    except Exception:
        from faster_whisper.audio import decode_audio

        return decode_audio(str(path), sampling_rate=target_sr).astype(np.float32)


def rms_dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + EPS)) if len(x) else -120.0


def frame_levels_db(x: np.ndarray, sr: int, frame_ms: float = 30.0, hop_ms: float = 10.0):
    """Short-time RMS in dBFS. Returns (levels, frame_len, hop) with sample counts."""
    frame, hop = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    if len(x) < frame:
        return np.array([rms_dbfs(x)]), frame, hop
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(np.square(x[idx], dtype=np.float64), axis=1))
    return 20 * np.log10(rms + EPS), frame, hop


def detect_speech(
    x: np.ndarray,
    sr: int = SAMPLE_RATE,
    *,
    rel_db: float = 30.0,
    floor_margin_db: float = 9.0,
    abs_floor_db: float = -55.0,
    hangover_s: float = 0.25,
    min_speech_s: float = 0.1,
) -> list[tuple[int, int]]:
    """Energy-based voice activity detection. Returns [(start, end)] in samples.

    A frame is speech when its level is above `max(peak - rel_db, noise floor + floor_margin_db,
    abs_floor_db)`, where the peak is the 95th and the floor the 10th percentile of frame levels.
    Gaps shorter than `hangover_s` are bridged and bursts shorter than `min_speech_s` dropped.
    Steady noise or a constant tone has no level contrast and is reported as "no speech".
    """
    if len(x) == 0:
        return []
    levels, frame, hop = frame_levels_db(x, sr)
    peak, floor = np.percentile(levels, 95), np.percentile(levels, 10)
    thr = max(peak - rel_db, floor + floor_margin_db, abs_floor_db)
    active = levels > thr
    if not active.any():
        return []

    runs, start = [], None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(active)])

    gap = int(hangover_s * sr / hop)
    merged = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    min_len = int(min_speech_s * sr / hop)
    return [
        (s * hop, min(len(x), (e - 1) * hop + frame))
        for s, e in merged
        if e - s >= min_len
    ]


def trim_silence(
    x: np.ndarray,
    segments: list[tuple[int, int]],
    sr: int = SAMPLE_RATE,
    *,
    max_pause_s: float = 0.5,
    margin_s: float = 0.1,
) -> np.ndarray:
    """Drop leading/trailing silence and shrink pauses longer than `max_pause_s`."""
    if not segments:
        return x[:0]
    pad = int(max(margin_s, max_pause_s / 2) * sr)
    spans = [[max(0, s - pad), min(len(x), e + pad)] for s, e in segments]
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return np.concatenate([x[s:e] for s, e in merged])


def normalise_level(x: np.ndarray, mode: str = "rms", target_dbfs: float = -26.0) -> np.ndarray:
    """Scale the waveform. `rms`: speech RMS to `target_dbfs`, `peak`: peak to 0.95, `none`: untouched.

    The output is limited to +-0.99 so that a loud peak never clips.
    """
    if mode == "none" or len(x) == 0:
        return x
    if mode == "peak":
        peak = float(np.max(np.abs(x)))
        return x * (0.95 / peak) if peak > EPS else x
    if mode == "rms":
        level = rms_dbfs(x)
        if level < -100:
            return x
        y = x * 10 ** ((target_dbfs - level) / 20)
        peak = float(np.max(np.abs(y)))
        return y * (0.99 / peak) if peak > 0.99 else y
    raise ValueError(f"unknown normalisation mode: {mode!r}")


def sliding_windows(
    x: np.ndarray, sr: int = SAMPLE_RATE, win_s: float = 6.0, hop_s: float = 3.0, min_tail_s: float = 1.0
) -> list[np.ndarray]:
    """Split a long signal into overlapping windows; a short signal stays whole.

    The last window is aligned on the end of the signal when more than `min_tail_s` of audio
    would otherwise be left out.
    """
    n, w, h = len(x), int(win_s * sr), int(hop_s * sr)
    if n <= w:
        return [x]
    starts = list(range(0, n - w + 1, h))
    if starts[-1] + w < n and n - (starts[-1] + w) >= int(min_tail_s * sr):
        starts.append(n - w)
    return [x[s : s + w] for s in starts]
