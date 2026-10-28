import numpy as np
import pytest

from emotionllm.evaluation import (
    choose_threshold,
    confusion_matrix,
    expected_calibration_error,
    fit_temperature,
    leave_one_session_out,
    report,
    selective_report,
    session_of,
    speaker_of,
)

LABELS = ["a", "b", "c"]


def test_ids():
    assert session_of("Ses03M_impro02_F000") == 3
    assert speaker_of("Ses03M_impro02_F000") == "Ses03F"
    assert speaker_of("Ses05F_script01_1_M012") == "Ses05M"


def test_confusion_matrix_rows_are_true_classes():
    cm = confusion_matrix([0, 0, 1, 2], [0, 1, 1, 2], 3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_report_metrics():
    y_true = [0, 0, 0, 0, 1, 2]
    y_pred = [0, 0, 0, 1, 1, 0]
    r = report(y_true, y_pred, LABELS)
    assert r["accuracy"] == pytest.approx(4 / 6)  # weighted accuracy
    assert r["balanced_accuracy"] == pytest.approx((3 / 4 + 1 + 0) / 3)  # unweighted accuracy
    assert r["per_class"]["c"]["recall"] == 0 and r["per_class"]["c"]["support"] == 1
    assert r["per_class"]["a"]["precision"] == pytest.approx(3 / 4)


def test_report_with_absent_class_does_not_divide_by_zero():
    r = report([0, 1], [0, 1], LABELS)
    assert r["balanced_accuracy"] == 1.0 and r["per_class"]["c"]["support"] == 0


def test_leave_one_session_out_partitions():
    sessions = [1, 1, 2, 3, 3, 3]
    folds = list(leave_one_session_out(sessions))
    assert [f[0] for f in folds] == [1, 2, 3]
    for s, train, test in folds:
        assert set(train).isdisjoint(test) and len(train) + len(test) == len(sessions)
        assert all(sessions[i] == s for i in test)


def test_selective_report_trades_coverage_for_accuracy():
    probs = np.array([[0.9, 0.05, 0.05], [0.4, 0.35, 0.25], [0.1, 0.8, 0.1], [0.34, 0.33, 0.33]])
    y = [0, 1, 1, 2]
    rows = selective_report(probs, y, [0.0, 0.5], fallback=0)
    assert rows[0]["coverage"] == 1.0 and rows[0]["accuracy"] == pytest.approx(2 / 4)
    assert rows[1]["coverage"] == 0.5 and rows[1]["committed_accuracy"] == 1.0


def test_choose_threshold_picks_the_smallest_one_meeting_the_target():
    rows = [
        {"threshold": 0.0, "coverage": 1.0, "committed_accuracy": 0.37},
        {"threshold": 0.3, "coverage": 0.6, "committed_accuracy": 0.43},
        {"threshold": 0.4, "coverage": 0.3, "committed_accuracy": 0.50},
        {"threshold": 0.5, "coverage": 0.2, "committed_accuracy": 0.58},
    ]
    assert choose_threshold(rows, 0.5) == 0.4
    assert choose_threshold(rows, 0.9) == 0.5  # unreachable target: the most selective threshold tested


def test_calibration_error_zero_when_confidence_matches_accuracy():
    probs = np.tile([0.8, 0.1, 0.1], (10, 1))
    y = [0] * 8 + [1, 2]
    assert expected_calibration_error(probs, y) == pytest.approx(0.0, abs=1e-9)
    assert expected_calibration_error(probs, [1] * 10) == pytest.approx(0.8)


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 3, n)
    logits = rng.normal(0, 1, (n, 3))
    logits[np.arange(n), y] += 1.0
    overconfident = logits * 4.0
    t = fit_temperature(overconfident, y)
    assert 3.0 < t < 5.0
    assert fit_temperature(logits, y) == pytest.approx(1.0, abs=0.25)
