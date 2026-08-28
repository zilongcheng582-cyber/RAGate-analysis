from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from figure2_spec import PAPER_FIGURE2_LABELS, PAPER_FIGURE2_ORDER  # noqa: E402

from make_paper_figures import (  # noqa: E402
    DATASETS,
    FIGURE3_SETTINGS,
    figure3_values,
    normalized_feature_importance,
)


def test_figure2_is_normalized_within_each_dataset() -> None:
    features, normalized = normalized_feature_importance(ROOT / "reference_results")
    expected_order = [
        "turn_position_ratio",
        "turn_position_squared",
        "user_has_question",
        "prev_sys_is_question",
        "user_starts_question_word",
        "user_turn_len_log",
        "sys_turn_len_log",
        "dialogue_len_log",
        "consecutive_sys_turns",
        "turn_len_ratio",
    ]
    assert PAPER_FIGURE2_ORDER == expected_order
    assert features == expected_order
    assert [PAPER_FIGURE2_LABELS[feature] for feature in features] == [
        "pos_ratio",
        "pos_sq",
        "user_q",
        "prev_sys_q",
        "user_q_word",
        "usr_len",
        "sys_len",
        "dlg_len",
        "cons_sys",
        "len_ratio",
    ]
    assert set(normalized) == set(DATASETS)
    for values in normalized.values():
        assert np.isclose(values.max(), 1.0)


def test_figure3_reads_three_csv_backed_series() -> None:
    values = figure3_values(ROOT / "reference_results")
    assert tuple(values) == ("Structural LR", "Embedding + LR", "Fine-tuned BERT")
    assert all(series.shape == (len(FIGURE3_SETTINGS),) for series in values.values())
    assert all(np.isfinite(series).all() for series in values.values())
