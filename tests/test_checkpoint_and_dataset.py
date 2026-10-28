import numpy as np
import torch

from emotionllm.dataset import IEMOCAP_CODES, Corpus, windowed_features
from emotionllm.features import N_FRAMES, N_MFCC, SAMPLE_RATE, cnn_input, mfcc
from emotionllm.inference import DEFAULT_THRESHOLD, EmotionClassifier
from emotionllm.model import LABELS, Checkpoint, EmotionClassificationCNN, load_checkpoint, load_model, save_checkpoint
from emotionllm.preprocess import LEGACY, PreprocessConfig

from .conftest import speech_like


def test_label_order_matches_iemocap_codes():
    assert [name[:3].lower() for name in LABELS] == list(IEMOCAP_CODES)


def test_bare_state_dict_loads_as_legacy(tmp_path):
    """The format of the original project's weight files: no statistics, original preprocessing."""
    path = tmp_path / "bare.pth"
    torch.save(EmotionClassificationCNN().state_dict(), path)
    ckpt = load_checkpoint(path)
    assert ckpt.stats is None and ckpt.preprocess is None and ckpt.temperature == 1.0
    clf = EmotionClassifier.from_weights(path)
    assert clf.config.preprocess == LEGACY and clf.config.threshold == DEFAULT_THRESHOLD
    assert isinstance(load_model(path), EmotionClassificationCNN)


def test_checkpoint_roundtrip_restores_the_recipe(tmp_path):
    model = EmotionClassificationCNN()
    mean, std = np.arange(N_MFCC, dtype=np.float32), np.full(N_MFCC, 2.0, dtype=np.float32)
    preprocess = {"normalise": "rms", "window_s": 8.0, "hop_s": 4.0}
    path = tmp_path / "ckpt.pth"
    save_checkpoint(path, Checkpoint(model, (mean, std), preprocess, temperature=1.7, threshold=0.55, meta={"k": "v"}))
    loaded = load_checkpoint(path)
    assert np.array_equal(loaded.stats[0], mean) and np.array_equal(loaded.stats[1], std)
    assert loaded.temperature == 1.7 and loaded.threshold == 0.55 and loaded.meta == {"k": "v"}
    for a, b in zip(model.state_dict().values(), loaded.model.state_dict().values()):
        assert torch.equal(a, b)
    clf = EmotionClassifier.from_weights(path)
    assert clf.config.preprocess == PreprocessConfig(normalise="rms", window_s=8.0, hop_s=4.0)
    assert clf.config.temperature == 1.7 and clf.config.threshold == 0.55
    assert clf.stats is not None
    assert clf.classify(speech_like(3.0)).n_windows == 1  # runs end to end with the restored recipe


def test_standardisation_happens_before_zero_padding():
    y = speech_like(2.0)
    raw = mfcc(y)
    mean, std = raw.mean(axis=1), raw.std(axis=1) + 1e-6
    x = cnn_input(y, SAMPLE_RATE, (mean, std))
    assert x.shape == (1, N_MFCC, N_FRAMES)
    assert abs(x[0, :, : raw.shape[1]].mean()) < 1e-3  # standardised frames
    assert np.all(x[0, :, raw.shape[1] :] == 0)  # padding = the average frame


def make_corpus(n=6):
    clips = [speech_like(3.0 + i, seed=i) for i in range(n)]
    parts = [(c * 32767).astype(np.int16) for c in clips]
    offsets = np.concatenate([[0], np.cumsum([len(p) for p in parts])])
    utt_ids = np.array([f"Ses0{1 + i % 5}F_impro01_F00{i}" for i in range(n)])
    return Corpus(np.concatenate(parts), offsets, np.arange(n) % 6, utt_ids, np.zeros(n, dtype=np.int8))


def test_corpus_accessors_and_subset():
    corpus = make_corpus()
    assert len(corpus) == 6 and corpus.sessions.tolist() == [1, 2, 3, 4, 5, 1]
    assert corpus.speakers[0] == "Ses01F"
    sub = corpus.subset([1, 4])
    assert len(sub) == 2 and np.array_equal(sub.waveform(0), corpus.waveform(1))
    assert sub.labels.tolist() == [1, 4]


def test_windowed_features_owners_and_shapes():
    corpus = make_corpus()
    feats, owners = windowed_features(corpus, [0, 5], PreprocessConfig(window_s=3.0, hop_s=2.0))
    assert feats.shape[1:] == (1, N_MFCC, N_FRAMES) and feats.shape[0] == len(owners)
    assert set(owners.tolist()) == {0, 1} and (owners == 1).sum() > 1  # the 8 s utterance is split
