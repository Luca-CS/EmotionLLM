"""Measure the two models of the original project on the IEMOCAP archive (see `prepare_iemocap.py`).

    python scripts/evaluate_original.py data/iemocap.npz \
        --cnn weights/original/emotion_classification_model_weight_2.pth \
        --transformer weights/original/emotion_classification_model_weight_3.pth --out results/original_models.json

The notebooks trained on the first 5,120 utterances of `train_set.pkl` and evaluated on the first 512
(CNN) or 160 (Transformer) utterances of `test_set.pkl`. Both splits are by utterance, not by speaker.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from legacy_models import EmotionClassificationTransformer  # noqa: E402

from emotionllm.dataset import legacy_features, load_corpus  # noqa: E402
from emotionllm.evaluation import report  # noqa: E402
from emotionllm.model import LABELS, EmotionClassificationCNN  # noqa: E402


@torch.no_grad()
def predict(model, features: np.ndarray, batch: int = 64) -> np.ndarray:
    model.eval()
    return np.concatenate([model(torch.from_numpy(features[i : i + batch])).numpy() for i in range(0, len(features), batch)])


def score(model, corpus, indices) -> dict:
    logits = predict(model, legacy_features(corpus, indices))
    pred = logits.argmax(axis=1)
    out = report(corpus.labels[indices], pred, LABELS)
    out["predicted_class_counts"] = {name: int((pred == k).sum()) for k, name in enumerate(LABELS)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--cnn", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/original_models.json"))
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    train = np.flatnonzero(corpus.original_split == 0)
    test = np.flatnonzero(corpus.original_split == 1)
    print(f"{len(train)} train / {len(test)} test utterances, sessions {sorted(set(corpus.sessions.tolist()))}")

    cnn = EmotionClassificationCNN()
    cnn.load_state_dict(torch.load(args.cnn, map_location="cpu", weights_only=True))
    transformer = EmotionClassificationTransformer()
    transformer.load_state_dict(torch.load(args.transformer, map_location="cpu", weights_only=True))

    runs = {
        "cnn_weight_2": {
            "train_first_5120 (notebook: 71.50 %)": (cnn, train[:5120]),
            "test_first_512 (notebook: 39.65 %)": (cnn, test[:512]),
            "test_all_1500": (cnn, test),
        },
        "transformer_weight_3": {
            "test_first_160 (notebook subset)": (transformer, test[:160]),
            "test_all_1500": (transformer, test),
        },
    }
    results = {}
    for model_name, subsets in runs.items():
        results[model_name] = {}
        for subset_name, (model, idx) in subsets.items():
            r = score(model, corpus, idx)
            results[model_name][subset_name] = r
            print(
                f"{model_name:22s} {subset_name:38s} n={r['n']:5d} acc={r['accuracy']:.4f} "
                f"UA={r['balanced_accuracy']:.4f} F1={r['macro_f1']:.4f} predicted={r['predicted_class_counts']}"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
