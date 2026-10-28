"""Retrain the CNN and measure it with a speaker-independent protocol (leave-one-session-out).

    python scripts/train_cnn.py data/iemocap.npz --variant legacy --out results/cnn_legacy.json
    python scripts/train_cnn.py data/iemocap.npz --variant robust --out results/cnn_robust.json \
        --export weights/emotion_cnn.pth

Variants (same network, different recipe):
  legacy  the original recipe: raw MFCCs, no silence removal, no normalisation, first 16 s, plain cross-entropy.
  robust  16 kHz RMS normalisation, voice activity detection, 8 s sliding windows, per-coefficient MFCC
          standardisation, class-weighted loss, light augmentation (speed, noise, time/frequency masking).

For each of the 5 IEMOCAP sessions: train on the 4 others (one of them is the validation set for early stopping),
test on the held-out one. Held-out predictions are pooled to fit the softmax temperature and to pick the confidence
threshold of the "Neutral" fallback. `--export` then trains on all 5 sessions and saves a checkpoint.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from emotionllm.audio import resample
from emotionllm.dataset import load_corpus
from emotionllm.evaluation import (
    choose_threshold,
    expected_calibration_error,
    fit_temperature,
    leave_one_session_out,
    report,
    selective_report,
)
from emotionllm.features import N_FRAMES, N_MFCC, SAMPLE_RATE, mfcc, pad_or_crop
from emotionllm.model import LABELS, Checkpoint, EmotionClassificationCNN, save_checkpoint
from emotionllm.preprocess import LEGACY, PreprocessConfig, prepare

NEUTRAL = LABELS.index("Neutral")
VARIANTS = {
    "legacy": dict(preprocess=LEGACY, cmvn=False, augment=False, class_weights=False),
    "robust": dict(preprocess=PreprocessConfig(window_s=8.0, hop_s=4.0), cmvn=True, augment=True, class_weights=True),
}


# ---------------------------------------------------------------- features
def augmented_copy(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One random perturbation: speed 0.9 or 1.1, then white noise at 10 to 25 dB SNR half of the time."""
    if rng.random() < 0.7:
        x = resample(x, 10, 9) if rng.random() < 0.5 else resample(x, 10, 11)  # slower / faster (and lower / higher)
    if rng.random() < 0.5:
        snr = rng.uniform(10, 25)
        noise = rng.standard_normal(len(x)).astype(np.float32)
        x = x + noise * np.sqrt(np.mean(x**2) / 10 ** (snr / 10) + 1e-12)
    return x.astype(np.float32)


def build_windows(corpus, cfg: PreprocessConfig, augment: bool, seed: int = 0) -> dict:
    """Raw (unstandardised, zero-padded) MFCC windows of every utterance, with an optional augmented copy."""
    rng = np.random.default_rng(seed)
    raw, n_frames, owner, is_aug = [], [], [], []
    for i in range(len(corpus)):
        x = corpus.waveform(i)
        for aug in (0, 1) if augment else (0,):
            prepared = prepare(augmented_copy(x, rng) if aug else x, SAMPLE_RATE, cfg)
            for window in prepared.windows:
                m = mfcc(window, SAMPLE_RATE)
                n_frames.append(min(m.shape[1], N_FRAMES))
                raw.append(pad_or_crop(m))
                owner.append(i)
                is_aug.append(aug)
        if i % 1000 == 0:
            print(f"  features {i}/{len(corpus)}", flush=True)
    return dict(
        raw=np.stack(raw).astype(np.float32),
        n_frames=np.array(n_frames),
        owner=np.array(owner),
        aug=np.array(is_aug),
    )


def mfcc_stats(w: dict, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of every coefficient over the valid frames of the given windows."""
    valid = np.arange(N_FRAMES)[None, :] < w["n_frames"][rows][:, None]
    frames = w["raw"][rows].transpose(0, 2, 1)[valid]  # (n_valid_frames, 13)
    return frames.mean(axis=0).astype(np.float32), (frames.std(axis=0) + 1e-6).astype(np.float32)


def make_batch(w, rows, stats, augment: bool, rng) -> torch.Tensor:
    x = (w["raw"][rows] - stats[0][None, :, None]) / stats[1][None, :, None]
    x = x * (np.arange(N_FRAMES)[None, :] < w["n_frames"][rows][:, None])[:, None, :]
    if augment:
        for b in range(len(x)):
            n = int(w["n_frames"][rows[b]])
            if rng.random() < 0.5:  # shift left by up to 30 frames
                k = int(rng.integers(0, 31))
                x[b, :, : N_FRAMES - k] = x[b, :, k:].copy()
                x[b, :, N_FRAMES - k :] = 0
            if rng.random() < 0.5 and n > 40:  # time mask
                width = int(rng.integers(5, 26))
                start = int(rng.integers(0, max(n - width, 1)))
                x[b, :, start : start + width] = 0
            if rng.random() < 0.3:  # frequency mask: one coefficient (not c0)
                x[b, int(rng.integers(1, N_MFCC))] = 0
            x[b, :, :n] += rng.normal(0, 0.05, (N_MFCC, n))
    return torch.from_numpy(x[:, None].astype(np.float32))


# ---------------------------------------------------------------- training
@torch.no_grad()
def window_logits(model, w, rows, stats, batch: int = 128) -> np.ndarray:
    model.eval()
    return np.concatenate(
        [model(make_batch(w, rows[i : i + batch], stats, False, None)).numpy() for i in range(0, len(rows), batch)]
    )


def utterance_probs(logits: np.ndarray, owners: np.ndarray, utts: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Average of the window softmaxes of each utterance, rows ordered like `utts`."""
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    out = np.zeros((len(utts), len(LABELS)))
    for k, u in enumerate(utts):
        sel = owners == u
        out[k] = p[sel].mean(axis=0) if sel.any() else np.eye(len(LABELS))[NEUTRAL]  # no speech: Neutral
    return out


def train(w, labels, train_rows, val_rows, variant, epochs, patience, seed):
    """Returns (model, stats, best_epoch). `val_rows=None`: train for exactly `epochs` epochs."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    stats = mfcc_stats(w, train_rows[w["aug"][train_rows] == 0]) if variant["cmvn"] else (
        np.zeros(N_MFCC, np.float32), np.ones(N_MFCC, np.float32))
    y_all = labels[w["owner"]]
    model = EmotionClassificationCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    counts = np.bincount(y_all[train_rows], minlength=len(LABELS)).astype(np.float64)
    weights = torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights if variant["class_weights"] else None, label_smoothing=0.05)
    best, best_state, best_epoch, stale = -1.0, None, epochs, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_rows)
        for i in range(0, len(order), 64):
            rows = order[i : i + 64]
            x = make_batch(w, rows, stats, variant["augment"], rng)
            loss = loss_fn(model(x), torch.from_numpy(y_all[rows]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if val_rows is None:
            continue
        val_utts = np.unique(w["owner"][val_rows])
        probs = utterance_probs(window_logits(model, w, val_rows, stats), w["owner"][val_rows], val_utts)
        ua = report(labels[val_utts], probs.argmax(axis=1), LABELS)["balanced_accuracy"]
        print(f"    epoch {epoch:2d} loss {loss.item():.3f} val UA {ua:.3f}", flush=True)
        if ua > best:
            best, best_epoch, stale = ua, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval(), stats, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--export", type=Path, help="also train on all sessions and save a checkpoint here")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-committed-accuracy", type=float, default=0.5, help="target for the Neutral-fallback threshold")
    parser.add_argument("--limit", type=int, help="use N utterances spread over the corpus only (smoke test)")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    variant = VARIANTS[args.variant]

    corpus = load_corpus(args.corpus)
    if args.limit:
        corpus = corpus.subset(np.linspace(0, len(corpus) - 1, args.limit).astype(int))
    labels, sessions = corpus.labels, corpus.sessions
    t0 = time.time()
    print(f"variant {args.variant}: {len(corpus)} utterances, building features", flush=True)
    w = build_windows(corpus, variant["preprocess"], variant["augment"])
    print(f"{len(w['raw'])} windows in {time.time() - t0:.0f}s", flush=True)
    window_session = sessions[w["owner"]]

    folds, pooled_logits, pooled_owner = [], [], []
    for held_out, _, _ in leave_one_session_out(sessions):
        val_session = held_out % 5 + 1
        train_rows = np.flatnonzero((window_session != held_out) & (window_session != val_session))
        val_rows = np.flatnonzero((window_session == val_session) & (w["aug"] == 0))
        test_rows = np.flatnonzero((window_session == held_out) & (w["aug"] == 0))
        print(f"fold: held-out session {held_out}, validation session {val_session}", flush=True)
        model, stats, best_epoch = train(w, labels, train_rows, val_rows, variant, args.epochs, args.patience, seed=held_out)
        logits = window_logits(model, w, test_rows, stats)
        test_utts = np.flatnonzero(sessions == held_out)
        probs = utterance_probs(logits, w["owner"][test_rows], test_utts)
        r = report(labels[test_utts], probs.argmax(axis=1), LABELS)
        r.update(held_out_session=int(held_out), best_epoch=best_epoch)
        folds.append(r)
        pooled_logits.append(logits)
        pooled_owner.append(w["owner"][test_rows])
        print(f"  -> session {held_out}: acc {r['accuracy']:.3f} UA {r['balanced_accuracy']:.3f} F1 {r['macro_f1']:.3f}", flush=True)

    logits, owners = np.concatenate(pooled_logits), np.concatenate(pooled_owner)
    all_utts = np.arange(len(corpus))
    probs = utterance_probs(logits, owners, all_utts)
    pooled = report(labels, probs.argmax(axis=1), LABELS)
    temperature = fit_temperature(logits, labels[owners])
    calibrated = utterance_probs(logits, owners, all_utts, temperature)
    thresholds = np.round(np.arange(0.0, 0.85, 0.05), 2)
    selective = selective_report(calibrated, labels, thresholds, NEUTRAL)
    best_threshold = choose_threshold(selective, args.min_committed_accuracy)
    summary = {
        "variant": args.variant,
        "recipe": {k: (dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v) for k, v in variant.items()},
        "per_session": folds,
        "mean_accuracy": float(np.mean([f["accuracy"] for f in folds])),
        "mean_balanced_accuracy": float(np.mean([f["balanced_accuracy"] for f in folds])),
        "std_balanced_accuracy": float(np.std([f["balanced_accuracy"] for f in folds])),
        "pooled": pooled,
        "temperature": temperature,
        "ece_before": expected_calibration_error(probs, labels),
        "ece_after": expected_calibration_error(calibrated, labels),
        "selective": selective,
        "chosen_threshold": best_threshold,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(
        f"LOSO pooled: acc {pooled['accuracy']:.3f} UA {pooled['balanced_accuracy']:.3f} F1 {pooled['macro_f1']:.3f} | "
        f"T={temperature:.2f} ECE {summary['ece_before']:.3f}->{summary['ece_after']:.3f} threshold {best_threshold}",
        flush=True,
    )

    if args.export:
        epochs = int(np.median([f["best_epoch"] for f in folds])) + 1
        print(f"final model on all sessions, {epochs} epochs", flush=True)
        rows = np.arange(len(w["raw"]))
        model, stats, _ = train(w, labels, rows, None, variant, epochs, args.patience, seed=0)
        preprocess = dataclasses.asdict(variant["preprocess"])
        meta = {
            "trained_on": "IEMOCAP, 5 sessions, 6 classes (7,380 utterances)",
            "protocol": "leave-one-session-out for the reported numbers, final model trained on all sessions",
            "loso_balanced_accuracy": summary["mean_balanced_accuracy"],
            "loso_accuracy": summary["mean_accuracy"],
            "epochs": epochs,
        }
        save_checkpoint(args.export, Checkpoint(model, stats if variant["cmvn"] else None, preprocess, temperature, best_threshold, meta))
        print(f"saved {args.export}")


if __name__ == "__main__":
    main()
