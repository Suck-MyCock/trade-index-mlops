"""Train the trade-index regression models and log them to MLflow.

Two experiments are produced, as the project brief requires:

    * ``baseline-linear``  a LinearRegression on the same features, giving a
      reference point for "what does a trivial model achieve?"
    * ``tuned-forest``     a RandomForestRegressor whose hyperparameters come
      from ``ml/configs/baseline.yaml``, showing the gain from a richer model.

Every run records the five things the brief asks for: git commit, DVC data
version, hyperparameters, metrics and the model artifact.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.data_pipeline import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

DEFAULT_CONFIG = Path("ml/configs/baseline.yaml")
RANDOM_STATE = 42

# skops refuses to load tree-based sklearn estimators unless their internal
# type is explicitly trusted. We build these models in this process, so the
# risk the warning describes (a malicious pickle) does not apply.
TRUSTED_SKOPS_TYPES = ["sklearn.tree._tree.Tree"]


def git_commit() -> str:
    """Return the current commit SHA, or ``"unknown"`` outside a git checkout."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def dvc_hash(pointer_path: str | Path) -> str:
    """Read the ``md5`` value out of a DVC pointer file.

    That hash is the dataset version identifier: it changes whenever the data
    changes, which is exactly what the brief means by "DVC dataset version ID".

    A DVC pointer looks like::

        outs:
        - md5: 42171ec985af1d3aed38f4d6f497030b
          size: 23324132
          path: raw_dataset.csv

    so the md5 line starts with a YAML list marker; strip it before matching.
    """
    path = Path(pointer_path)
    if not path.exists():
        return "unknown"
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        if stripped.startswith("md5:"):
            return stripped.split("md5:", 1)[1].strip()
    return "unknown"


def load_config(path: str | Path) -> dict:
    """Load the YAML hyperparameter file."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def available_categoricals(columns: list[str]) -> list[str]:
    """Return the categorical features actually present in the dataset.

    v2 drops columns that collapse to a single value (``MAGNITUDE`` after the
    Index-unit filter), so the training script must not assume every name in
    :data:`CATEGORICAL_FEATURES` survives.
    """
    return [c for c in CATEGORICAL_FEATURES if c in columns]


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read the v2 CSV and split it into features and target."""
    frame = pd.read_csv(path)

    required = NUMERIC_FEATURES + [TARGET]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")

    categoricals = available_categoricals(list(frame.columns))
    if not categoricals:
        raise ValueError("dataset has no categorical feature columns")

    return frame[NUMERIC_FEATURES + categoricals], frame[TARGET]


def build_preprocessor(categoricals: list[str]) -> ColumnTransformer:
    """One-hot encode categoricals, pass numeric features straight through.

    ``handle_unknown="ignore"`` matters for the API: a category never seen in
    training produces an all-zero block instead of raising.
    """
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricals),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the regression metrics we log for every run."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_experiment(
    name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: dict,
    categoricals: list[str],
    commit: str,
    data_version: str,
    data_hash: str,
) -> dict[str, float]:
    """Fit one model, evaluate it, and record everything in MLflow."""
    pipeline = Pipeline(
        steps=[("prep", build_preprocessor(categoricals)), ("model", model)]
    )

    with mlflow.start_run(run_name=name):
        mlflow.log_param("git_commit", commit)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("data_dvc_hash", data_hash)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        for key, value in params.items():
            mlflow.log_param(key, value)

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = evaluate(y_test, predictions)

        mlflow.log_metrics(metrics)
        # MLflow 3.x serialises sklearn models with skops, which refuses tree
        # ensembles unless the tree type is explicitly trusted. We trained this
        # model ourselves in-process, so it is safe to trust it here.
        mlflow.sklearn.log_model(
            pipeline, name="model", skops_trusted_types=TRUSTED_SKOPS_TYPES
        )

        print(f"[{name}] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and log trade-index models.")
    parser.add_argument("--data", default="data/processed/clean_v2.csv")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--pointer", default="data/raw/raw_dataset.csv.dvc")
    parser.add_argument("--data-version", default="v2")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="trade-index")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    config = load_config(args.config)

    X, y = load_dataset(args.data)
    categoricals = available_categoricals(list(X.columns))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE
    )

    commit = git_commit()
    data_hash = dvc_hash(args.pointer)
    print(f"git commit : {commit}")
    print(f"data hash  : {data_hash}")
    print(f"train/test : {len(X_train)}/{len(X_test)}")
    print(f"categorical: {len(categoricals)}  -> encoded from v2 columns")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    common = dict(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        categoricals=categoricals,
        commit=commit,
        data_version=args.data_version,
        data_hash=data_hash,
    )

    baseline_params = {"model_type": "LinearRegression", "fit_intercept": True}
    run_experiment(
        name="baseline-linear",
        model=LinearRegression(fit_intercept=True),
        params=baseline_params,
        **common,
    )

    forest_cfg = config.get("random_forest", {})
    forest_params = {
        "model_type": "RandomForestRegressor",
        "n_estimators": forest_cfg.get("n_estimators", 100),
        "max_depth": forest_cfg.get("max_depth", 20),
        "min_samples_leaf": forest_cfg.get("min_samples_leaf", 1),
        "random_state": RANDOM_STATE,
    }
    run_experiment(
        name="tuned-forest",
        model=RandomForestRegressor(
            n_estimators=forest_params["n_estimators"],
            max_depth=forest_params["max_depth"],
            min_samples_leaf=forest_params["min_samples_leaf"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        params=forest_params,
        **common,
    )

    summary = {"git_commit": commit, "data_version": args.data_version,
               "data_dvc_hash": data_hash}
    Path("models").mkdir(exist_ok=True)
    Path("models/run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nRun ``mlflow ui --backend-store-uri {args.tracking_uri}`` to inspect.")


if __name__ == "__main__":
    main()
