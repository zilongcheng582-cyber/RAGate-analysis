from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
C_GRID = [0.01, 0.1, 1.0, 10.0]
CV_FOLDS = 5
N_SHUFFLE_REPEATS = 30
LABEL_COL = "label"
DATASET_ORDER = ["KETOD", "DSTC9", "DSTC11"]

ALL_FEATURES = [
    "turn_position_ratio",
    "prev_sys_is_question",
    "user_has_question",
    "user_starts_question_word",
    "user_turn_len_log",
    "sys_turn_len_log",
    "dialogue_len_log",
    "consecutive_sys_turns",
    "turn_len_ratio",
    "turn_position_squared",
]
POSITION_FEATURES = ["turn_position_ratio", "turn_position_squared"]


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        return pd.read_csv(path, engine="python", on_bad_lines="error")


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train_position_model(train_df: pd.DataFrame) -> tuple[Pipeline, float]:
    x = train_df[ALL_FEATURES].to_numpy(float)
    y = train_df[LABEL_COL].to_numpy(int)
    search = GridSearchCV(
        build_pipeline(),
        {"lr__C": C_GRID},
        cv=StratifiedKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
        ),
        scoring="f1_macro",
        n_jobs=1,
    )
    search.fit(x, y)
    return search.best_estimator_, float(search.best_params_["lr__C"])


def class_conditional_question_rates(
    config: dict[str, dict[str, str]], output_dir: Path
) -> pd.DataFrame:
    """Use the authoritative extracted current-turn feature, not raw context text."""
    rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        feature_path = Path(config[dataset]["train_features"])
        df = read_csv(feature_path)
        label = pd.to_numeric(df[LABEL_COL], errors="raise").astype(int)
        question = pd.to_numeric(
            df["user_has_question"], errors="raise"
        ).astype(int)
        if not set(question.unique()).issubset({0, 1}):
            raise ValueError(f"{dataset}: user_has_question is not binary")
        pos = question[label == 1]
        neg = question[label == 0]
        rows.append(
            {
                "dataset": dataset,
                "source_file": str(feature_path),
                "feature_definition": "current_user_turn",
                "N_pos": int(len(pos)),
                "N_neg": int(len(neg)),
                "pos_q_rate": float(pos.mean()),
                "neg_q_rate": float(neg.mean()),
                "delta": float(pos.mean() - neg.mean()),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(
        output_dir / "class_conditional_qrate_fixed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result


def joint_position_permutation(
    config: dict[str, dict[str, str]], output_dir: Path
) -> pd.DataFrame:
    """Jointly permute the two deterministic position columns by one row index."""
    rows: list[dict[str, object]] = []
    position_idx = [ALL_FEATURES.index(name) for name in POSITION_FEATURES]

    for dataset_index, dataset in enumerate(DATASET_ORDER):
        train_df = read_csv(Path(config[dataset]["train_features"]))
        test_df = read_csv(Path(config[dataset]["test_features"]))
        x_train = train_df[ALL_FEATURES].to_numpy(float)
        y_train = train_df[LABEL_COL].to_numpy(int)
        x_test = test_df[ALL_FEATURES].to_numpy(float)
        y_test = test_df[LABEL_COL].to_numpy(int)

        relation_error = np.max(
            np.abs(
                x_test[:, position_idx[1]]
                - np.square(x_test[:, position_idx[0]])
            )
        )
        if relation_error > 1e-12:
            raise ValueError(
                f"{dataset}: position-square relation is already inconsistent "
                f"(max error={relation_error})"
            )

        model, best_c = train_position_model(train_df)
        baseline = float(
            f1_score(y_test, model.predict(x_test), average="macro")
        )

        rng = np.random.default_rng(RANDOM_STATE + dataset_index)
        shuffled_scores: list[float] = []
        max_shuffled_relation_error = 0.0
        for _ in range(N_SHUFFLE_REPEATS):
            permutation = rng.permutation(len(x_test))
            shuffled = x_test.copy()
            shuffled[:, position_idx] = x_test[permutation][:, position_idx]
            max_shuffled_relation_error = max(
                max_shuffled_relation_error,
                float(
                    np.max(
                        np.abs(
                            shuffled[:, position_idx[1]]
                            - np.square(shuffled[:, position_idx[0]])
                        )
                    )
                ),
            )
            shuffled_scores.append(
                float(
                    f1_score(
                        y_test, model.predict(shuffled), average="macro"
                    )
                )
            )

        shuffled_mean = float(np.mean(shuffled_scores))
        rows.append(
            {
                "dataset": dataset,
                "shuffled_cols": ";".join(POSITION_FEATURES),
                "permutation_unit": "joint_row",
                "n_repeats": N_SHUFFLE_REPEATS,
                "best_C": best_c,
                "baseline_f1": baseline,
                "shuffled_mean": shuffled_mean,
                "shuffled_std": float(np.std(shuffled_scores)),
                "delta": baseline - shuffled_mean,
                "max_relation_error_after_shuffle": max_shuffled_relation_error,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(
        output_dir / "position_shuffle_group_fixed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config_paths.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))

    qrate = class_conditional_question_rates(config, args.output_dir)
    position = joint_position_permutation(config, args.output_dir)
    evidence_lines = [
        "# Corrected lightweight experiment evidence",
        "",
        "## Class-conditional current-turn question rate",
        "",
        "| Dataset | P(q|pos) | P(q|neg) | Delta |",
        "|---|---:|---:|---:|",
    ]
    for _, row in qrate.iterrows():
        evidence_lines.append(
            f"| {row['dataset']} | {row['pos_q_rate']:.3f} | "
            f"{row['neg_q_rate']:.3f} | {row['delta']:+.3f} |"
        )
    evidence_lines.extend(
        [
            "",
            "## Grouped position permutation",
            "",
            "| Dataset | Baseline Macro F1 | Shuffled mean | Std. | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in position.iterrows():
        evidence_lines.append(
            f"| {row['dataset']} | {row['baseline_f1']:.4f} | "
            f"{row['shuffled_mean']:.4f} | {row['shuffled_std']:.4f} | "
            f"{row['delta']:+.4f} |"
        )
    evidence_lines.extend(
        [
            "",
            "Question rates come from the authoritative current-user-turn feature column. "
            "The two position columns are jointly permuted by one row index, and their "
            "deterministic square relation is preserved.",
        ]
    )
    (args.output_dir / "lightweight_fix_evidence.md").write_text(
        "\n".join(evidence_lines), encoding="utf-8"
    )
    print("Corrected class-conditional question rates")
    print(qrate.to_string(index=False))
    print("\nCorrected grouped position permutation")
    print(position.to_string(index=False))


if __name__ == "__main__":
    main()
