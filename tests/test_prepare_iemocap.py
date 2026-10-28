import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_iemocap  # noqa: E402


def make_items():
    rng = np.random.default_rng(0)
    items = []
    for i, n in enumerate((16000, 24000)):
        raw = rng.integers(-3000, 3000, n).astype(np.float32) / 32768.0
        items.append((torch.from_numpy(raw)[None], 16000, "neu" if i == 0 else "sad", f"Ses0{i + 1}F_impro01_F00{i}", i))
    return items


def test_roundtrip_through_restricted_unpickler(tmp_path):
    items = make_items()
    for name in ("a.pkl", "b.pkl"):
        with open(tmp_path / name, "wb") as f:
            pickle.dump(items, f)
    out = tmp_path / "out.npz"
    prepare_iemocap.convert([tmp_path / "a.pkl", tmp_path / "b.pkl"], out)
    data = np.load(out)
    assert len(data["labels"]) == 4 and list(data["original_split"]) == [0, 0, 1, 1]
    first = data["audio"][data["offsets"][0] : data["offsets"][1]]
    assert np.array_equal(first.astype(np.float32) / 32768.0, items[0][0][0].numpy())
    assert data["utt_ids"][1] == "Ses02F_impro01_F001"


def test_malicious_pickle_is_blocked(tmp_path):
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            return (os.system, (f"echo x > {marker}",))

    with open(tmp_path / "evil.pkl", "wb") as f:
        pickle.dump([Evil()], f)
    with pytest.raises(pickle.UnpicklingError, match="blocked global"):
        prepare_iemocap.load_pickle(tmp_path / "evil.pkl")
    assert not marker.exists()


def test_wrong_sample_rate_is_rejected(tmp_path):
    bad = [(torch.zeros(1, 100), 8000, "neu", "Ses01F_impro01_F000", 0)]
    with open(tmp_path / "bad.pkl", "wb") as f:
        pickle.dump(bad, f)
    with pytest.raises(ValueError, match="16 kHz"):
        prepare_iemocap.convert([tmp_path / "bad.pkl"], tmp_path / "o.npz")
