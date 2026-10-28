"""Figures of the README from the JSON reports in `results/`.

    python scripts/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["Angry", "Excited", "Frustrated", "Happy", "Neutral", "Sad"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#9cc0ee", "#2a78d6", "#12437f"])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10,
    }
)


def confusion(ax, report: dict, title: str) -> None:
    cm = np.array(report["confusion"], dtype=float)
    recall = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(recall, cmap=SEQ, vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{recall[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if recall[i, j] > 0.5 else INK)
    ax.set_xticks(range(6), LABELS, rotation=35, ha="right")
    ax.set_yticks(range(6), [f"{name} ({int(n)})" for name, n in zip(LABELS, cm.sum(axis=1))])
    ax.set_xlabel("predicted")
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)


def confusion_figure(original: dict, retrained: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.9))
    o = original["cnn_weight_2"]["test_all_1500"]
    confusion(axes[0], o, f"Original CNN, utterance-level test split (n={o['n']})\n"
                          f"accuracy {o['accuracy']:.0%}, mean recall {o['balanced_accuracy']:.0%}")
    axes[0].set_ylabel("true class (utterances), cell = share of the row")
    p = retrained["pooled"]
    confusion(axes[1], p, f"Retrained CNN, leave-one-session-out (n={p['n']})\n"
                          f"accuracy {p['accuracy']:.0%}, mean recall {p['balanced_accuracy']:.0%}")
    fig.tight_layout()
    fig.savefig(out, dpi=140)


def selective_figure(retrained: dict, out: Path) -> None:
    rows = retrained["selective"]
    t = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(t, [r["coverage"] for r in rows], color=ORANGE, lw=2, label="answers kept (coverage)")
    ax.plot(t, [r["committed_accuracy"] for r in rows], color=BLUE, lw=2, label="accuracy on the answers kept")
    ax.axvline(retrained["chosen_threshold"], color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.text(retrained["chosen_threshold"] + 0.01, 0.04, f"threshold {retrained['chosen_threshold']:.2f}", color=INK_2, fontsize=9)
    ax.legend(loc="upper right", frameon=False, labelcolor=INK_2, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("confidence threshold (below it the answer falls back to Neutral)")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(out, dpi=140)


if __name__ == "__main__":
    original = json.loads((ROOT / "results" / "original_models.json").read_text())
    retrained = json.loads((ROOT / "results" / "cnn_robust.json").read_text())
    (ROOT / "docs").mkdir(exist_ok=True)
    confusion_figure(original, retrained, ROOT / "docs" / "confusion.png")
    selective_figure(retrained, ROOT / "docs" / "fallback.png")
    print("wrote docs/confusion.png and docs/fallback.png")
