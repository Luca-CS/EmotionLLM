"""Metrics and splits used to measure the classifier (numpy only).

IEMOCAP has 5 sessions with 2 speakers each (the session number is in the utterance id, `Ses01F_...`).
Utterances of one speaker share voice, microphone and scripted dialogue, so a random utterance-level split
leaks speakers into the test set: the honest protocol is to hold out whole sessions.
"""

from collections.abc import Iterator, Sequence

import numpy as np


def session_of(utterance_id: str) -> int:
    """`Ses03M_impro02_F000` -> 3."""
    return int(utterance_id[3:5])


def speaker_of(utterance_id: str) -> str:
    """`Ses03M_impro02_F000` -> `Ses03F`: session number plus the gender letter that starts the last part."""
    return utterance_id[:5] + utterance_id.rsplit("_", 1)[-1][0]


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    """Rows: true class, columns: predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def report(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str]) -> dict:
    """Weighted accuracy (plain accuracy), unweighted accuracy (mean per-class recall), macro-F1, per-class scores."""
    cm = confusion_matrix(y_true, y_pred, len(labels))
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    recall = np.divide(tp, support, out=np.zeros(len(labels)), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(len(labels)), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(labels)), where=(precision + recall) > 0)
    present = support > 0
    return {
        "n": int(cm.sum()),
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),  # "weighted accuracy" (WA)
        "balanced_accuracy": float(recall[present].mean()) if present.any() else 0.0,  # "unweighted accuracy" (UA)
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class": {
            name: {"recall": float(recall[i]), "precision": float(precision[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(labels)
        },
        "confusion": cm.tolist(),
    }


def leave_one_session_out(sessions: Sequence[int]) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (held-out session, train indices, test indices) for every session."""
    sessions = np.asarray(sessions)
    for s in sorted(set(sessions.tolist())):
        yield s, np.flatnonzero(sessions != s), np.flatnonzero(sessions == s)


def selective_report(probs: np.ndarray, y_true: Sequence[int], thresholds: Sequence[float], fallback: int) -> list[dict]:
    """For each confidence threshold: answer `fallback` when max prob < threshold.

    `coverage` is the share of utterances where the model commits to its own guess, `committed_accuracy` its
    accuracy on those, `accuracy` the overall accuracy once hesitations are replaced by the fallback class.
    """
    probs, y_true = np.asarray(probs), np.asarray(y_true)
    top, conf = probs.argmax(axis=1), probs.max(axis=1)
    rows = []
    for t in thresholds:
        committed = conf >= t
        pred = np.where(committed, top, fallback)
        rows.append(
            {
                "threshold": float(t),
                "coverage": float(committed.mean()),
                "committed_accuracy": float((top[committed] == y_true[committed]).mean()) if committed.any() else 0.0,
                "accuracy": float((pred == y_true).mean()),
            }
        )
    return rows


def choose_threshold(selective: list[dict], min_committed_accuracy: float = 0.5) -> float:
    """Smallest threshold at which the answers the model commits to are right at least `min_committed_accuracy`.

    Maximising overall accuracy would be meaningless here (a threshold near 1/6 never triggers the fallback), so
    the rule fixes the quality of the emotions that are allowed to change the tone of the reply, and the
    coverage that results is reported next to it. Falls back to the highest threshold tested.
    """
    for row in selective:
        if row["coverage"] > 0 and row["committed_accuracy"] >= min_committed_accuracy:
            return row["threshold"]
    return selective[-1]["threshold"]


def expected_calibration_error(probs: np.ndarray, y_true: Sequence[int], n_bins: int = 10) -> float:
    probs, y_true = np.asarray(probs), np.asarray(y_true)
    conf, correct = probs.max(axis=1), probs.argmax(axis=1) == y_true
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def _nll(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(y)), y].mean())


def fit_temperature(logits: np.ndarray, y_true: Sequence[int]) -> float:
    """Temperature minimising the negative log-likelihood (golden-section search on [0.05, 10])."""
    logits, y = np.asarray(logits, dtype=np.float64), np.asarray(y_true)
    lo, hi, phi = 0.05, 10.0, (np.sqrt(5) - 1) / 2
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    for _ in range(60):
        if _nll(logits, y, a) < _nll(logits, y, b):
            hi, b = b, a
            a = hi - phi * (hi - lo)
        else:
            lo, a = a, b
            b = lo + phi * (hi - lo)
    return float((lo + hi) / 2)
