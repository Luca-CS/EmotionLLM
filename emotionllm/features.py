"""MFCC front-end of the CNN.

The weights were trained on `librosa.feature.mfcc(y, sr=16000, n_mfcc=13)` with the default
parameters, so this module reproduces that computation with numpy only (no librosa, hence no
numba at runtime). `tests/test_features.py` checks the output against librosa.
"""

from functools import lru_cache

import numpy as np

SAMPLE_RATE = 16_000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 13
N_FRAMES = 500  # input width of the CNN: 500 * 512 / 16000 = 16 s
TOP_DB = 80.0
AMIN = 1e-10


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Slaney mel scale: linear below 1 kHz, logarithmic above."""
    hz = np.asarray(hz, dtype=np.float64)
    f_sp = 200.0 / 3
    mel = hz / f_sp
    min_log_hz, logstep = 1000.0, np.log(6.4) / 27.0
    above = hz >= min_log_hz
    mel = np.where(above, min_log_hz / f_sp + np.log(np.maximum(hz, 1e-12) / min_log_hz) / logstep, mel)
    return mel


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    f_sp = 200.0 / 3
    hz = f_sp * mel
    min_log_mel, logstep = 1000.0 / f_sp, np.log(6.4) / 27.0
    above = mel >= min_log_mel
    return np.where(above, 1000.0 * np.exp(logstep * (mel - min_log_mel)), hz)


@lru_cache(maxsize=4)
def mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    """Slaney-normalised triangular filters, shape (n_mels, 1 + n_fft // 2)."""
    fft_freqs = np.linspace(0.0, sr / 2, 1 + n_fft // 2)
    mel_pts = np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2), n_mels + 2)
    mel_f = _mel_to_hz(mel_pts)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights.astype(np.float32)


@lru_cache(maxsize=4)
def _dct_basis(n_mfcc: int = N_MFCC, n_mels: int = N_MELS) -> np.ndarray:
    """Orthonormal DCT-II matrix keeping the first `n_mfcc` rows."""
    n = np.arange(n_mels)
    k = np.arange(n_mfcc)[:, None]
    basis = np.cos(np.pi * (2 * n + 1) * k / (2 * n_mels)) * np.sqrt(2.0 / n_mels)
    basis[0] *= 1.0 / np.sqrt(2.0)
    return basis.astype(np.float32)


def power_spectrogram(y: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> np.ndarray:
    """|STFT|^2 with a centred periodic Hann window, shape (1 + n_fft // 2, frames)."""
    y = np.asarray(y, dtype=np.float32)
    y = np.pad(y, n_fft // 2, mode="constant")
    n_frames = 1 + (len(y) - n_fft) // hop_length
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)
    idx = np.arange(n_fft)[None, :] + hop_length * np.arange(n_frames)[:, None]
    spec = np.fft.rfft(y[idx] * window, axis=1)
    return (spec.real**2 + spec.imag**2).T.astype(np.float32)


def mfcc(y: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    """MFCCs of a mono waveform, shape (n_mfcc, frames), float32."""
    if sr != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sr} Hz (resample first)")
    mel = mel_filterbank(sr) @ power_spectrogram(y)
    log_mel = 10.0 * np.log10(np.maximum(AMIN, mel))
    log_mel = np.maximum(log_mel, log_mel.max() - TOP_DB)
    return (_dct_basis(n_mfcc) @ log_mel).astype(np.float32)


def pad_or_crop(features: np.ndarray, n_frames: int = N_FRAMES) -> np.ndarray:
    """Zero-pad on the right (as in training) or truncate to a fixed number of frames."""
    cur = features.shape[-1]
    if cur >= n_frames:
        return features[..., :n_frames]
    pad = [(0, 0)] * (features.ndim - 1) + [(0, n_frames - cur)]
    return np.pad(features, pad)


def standardise(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-coefficient mean/variance normalisation of MFCCs shaped (n_mfcc, frames)."""
    return ((features - mean[:, None]) / std[:, None]).astype(np.float32)


def cnn_input(waveform: np.ndarray, sr: int = SAMPLE_RATE, stats: tuple | None = None) -> np.ndarray:
    """One waveform -> CNN input of shape (1, N_MFCC, N_FRAMES).

    `stats=(mean, std)` standardises the MFCCs before the zero padding, so that padding means "average frame".
    """
    m = mfcc(waveform, sr)
    if stats is not None:
        m = standardise(m, *stats)
    return pad_or_crop(m)[None]
