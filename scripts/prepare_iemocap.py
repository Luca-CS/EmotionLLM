"""Convert the IEMOCAP pickles of the original project into a compact `.npz`, without trusting the pickle.

IEMOCAP is distributed by USC under a license agreement: the data stays on your machine and is never committed
(`data/` and `*.npz` are in `.gitignore`).

The original notebooks pickled lists of `(waveform, sample_rate, label, utterance_id, number)` tuples, with
`waveform` a (1, n) float tensor. A pickle can run arbitrary code when loaded, so this script only resolves the few
globals needed to rebuild tensors and refuses everything else.

    python scripts/prepare_iemocap.py data/train_set.pkl data/test_set.pkl --out data/iemocap.npz
"""

import argparse
import io
import pickle
from pathlib import Path

import numpy as np
import torch

ALLOWED = {
    ("collections", "OrderedDict"),
    ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_tensor"),
    ("torch", "Size"),
    ("torch", "FloatStorage"),
    ("torch.storage", "UntypedStorage"),
}


def _storage_from_bytes(blob: bytes):
    """Replacement for `torch.storage._load_from_bytes`, which would call `torch.load` without `weights_only`."""
    return torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return _storage_from_bytes
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load_pickle(path: Path) -> list:
    with open(path, "rb") as f:
        return RestrictedUnpickler(f).load()


def to_int16(waveform: torch.Tensor) -> np.ndarray:
    x = waveform.reshape(-1).numpy().astype(np.float64) * 32768.0
    return np.clip(np.rint(x), -32768, 32767).astype(np.int16)


def convert(paths: list[Path], out: Path) -> None:
    audio, lengths, labels, utt_ids, splits = [], [], [], [], []
    worst = 0.0
    for split, path in enumerate(paths):
        items = load_pickle(path)
        print(f"{path.name}: {len(items)} utterances")
        for i in range(len(items)):
            waveform, sr, label, utt_id, _ = items[i]
            items[i] = None  # free memory as we go
            if sr != 16_000:
                raise ValueError(f"{utt_id}: expected 16 kHz audio, got {sr} Hz")
            x = to_int16(waveform)
            worst = max(worst, float(np.abs(waveform.reshape(-1).numpy() * 32768.0 - x).max()))
            audio.append(x)
            lengths.append(len(x))
            labels.append(label)
            utt_ids.append(utt_id)
            splits.append(split)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    np.savez(
        out,
        audio=np.concatenate(audio),
        offsets=offsets,
        labels=np.array(labels),
        utt_ids=np.array(utt_ids),
        original_split=np.array(splits, dtype=np.int8),
    )
    print(f"wrote {out}: {len(labels)} utterances, {offsets[-1] / 16000 / 3600:.2f} h, max int16 rounding {worst:.3f} LSB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pickles", nargs="+", type=Path, help="train_set.pkl test_set.pkl ...")
    parser.add_argument("--out", type=Path, default=Path("data/iemocap.npz"))
    args = parser.parse_args()
    convert(args.pickles, args.out)
