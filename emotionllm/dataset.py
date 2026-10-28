"""Loading the IEMOCAP archive produced by `scripts/prepare_iemocap.py` and turning it into CNN inputs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .evaluation import session_of, speaker_of
from .features import N_FRAMES, N_MFCC, SAMPLE_RATE, mfcc, pad_or_crop
from .preprocess import PreprocessConfig, prepare

IEMOCAP_CODES = ("ang", "exc", "fru", "hap", "neu", "sad")  # same order as `LABELS`


@dataclass
class Corpus:
    audio: np.ndarray  # int16, all utterances concatenated
    offsets: np.ndarray  # utterance i is audio[offsets[i]:offsets[i + 1]]
    labels: np.ndarray  # class index into `LABELS`
    utt_ids: np.ndarray
    original_split: np.ndarray  # 0 = train_set.pkl of the original project, 1 = test_set.pkl

    def __len__(self) -> int:
        return len(self.labels)

    def waveform(self, i: int) -> np.ndarray:
        return self.audio[self.offsets[i] : self.offsets[i + 1]].astype(np.float32) / 32768.0

    def subset(self, indices) -> "Corpus":
        parts = [self.audio[self.offsets[i] : self.offsets[i + 1]] for i in indices]
        offsets = np.concatenate([[0], np.cumsum([len(p) for p in parts])]).astype(np.int64)
        return Corpus(
            np.concatenate(parts), offsets, self.labels[indices], self.utt_ids[indices], self.original_split[indices]
        )

    @property
    def sessions(self) -> np.ndarray:
        return np.array([session_of(u) for u in self.utt_ids])

    @property
    def speakers(self) -> np.ndarray:
        return np.array([speaker_of(u) for u in self.utt_ids])


def load_corpus(path: str | Path) -> Corpus:
    data = np.load(path)
    labels = np.array([IEMOCAP_CODES.index(code) for code in data["labels"]])
    return Corpus(data["audio"], data["offsets"], labels, data["utt_ids"], data["original_split"])


def legacy_features(corpus: Corpus, indices) -> np.ndarray:
    """Exactly what the original notebooks fed the CNN: raw MFCCs, zero-padded or cut to 500 frames."""
    return np.stack([pad_or_crop(mfcc(corpus.waveform(i), SAMPLE_RATE)) for i in indices])[:, None]


def windowed_features(corpus: Corpus, indices, cfg: PreprocessConfig):
    """Preprocessed windows of the selected utterances.

    Returns (features of shape (n_windows, 1, 13, 500), owner index of every window in `indices`).
    Utterances without detected speech yield no window.
    """
    feats, owners = [], []
    for k, i in enumerate(indices):
        prepared = prepare(corpus.waveform(i), SAMPLE_RATE, cfg)
        for window in prepared.windows:
            feats.append(pad_or_crop(mfcc(window, SAMPLE_RATE)))
            owners.append(k)
    if not feats:
        return np.zeros((0, 1, N_MFCC, N_FRAMES), dtype=np.float32), np.zeros(0, dtype=np.int64)
    return np.stack(feats)[:, None], np.array(owners)
