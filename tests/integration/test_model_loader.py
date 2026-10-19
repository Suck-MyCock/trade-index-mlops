"""Boundary-case tests for the model holder.

Group 2 of 3. These exercise the loading and prediction layer without going
through HTTP, so a failure here points at the loader rather than the routing.
"""

from __future__ import annotations

import joblib
import pandas as pd
import pytest
from app.model_loader import (
    ModelHolder,
    ModelNotLoadedError,
)
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICALS = [
    "STATUS",
    "Subject",
    "Group",
    "Series_title_1",
    "Series_title_2",
    "Series_title_3",
    "Series_title_4",
]
NUMERICS = ["year", "month"]


def _row(**overrides) -> pd.DataFrame:
    base = {
        "year": 2024,
        "month": 6,
        "STATUS": "FINAL",
        "Subject": "Overseas Trade Indexes - Prices - OTP",
        "Group": "Export price indexes",
        "Series_title_1": "Price",
        "Series_title_2": "BEEF AND VEAL",
        "Series_title_3": "June 2002 Quarter Base (=1000)",
        "Series_title_4": "Unknown",
    }
    base.update(overrides)
    return pd.DataFrame([base])


@pytest.fixture()
def tiny_model(tmp_path):
    """A real sklearn pipeline fitted on three rows, saved to a temp file."""
    frame = pd.concat([_row(year=y) for y in (2000, 2001, 2002)], ignore_index=True)
    frame["target"] = [100.0, 200.0, 300.0]

    pipeline = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            CATEGORICALS,
                        ),
                        ("num", "passthrough", NUMERICS),
                    ]
                ),
            ),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )
    pipeline.fit(frame[NUMERICS + CATEGORICALS], frame["target"])

    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    return path


def test_load_missing_artifact_raises_actionable_error(tmp_path):
    holder = ModelHolder()
    missing = tmp_path / "nope.joblib"
    with pytest.raises(FileNotFoundError) as excinfo:
        holder.load(missing, summary_path=None)
    assert "ml.train" in str(excinfo.value)


def test_predict_before_load_raises():
    holder = ModelHolder()
    with pytest.raises(ModelNotLoadedError):
        holder.predict(_row())


def test_load_then_predict_returns_float(tiny_model):
    holder = ModelHolder()
    holder.load(tiny_model, summary_path=None)
    assert holder.is_loaded is True
    result = holder.predict(_row())
    assert isinstance(result, float)


def test_predict_rejects_multi_row_input(tiny_model):
    holder = ModelHolder()
    holder.load(tiny_model, summary_path=None)
    frame = pd.concat([_row(), _row(year=1999)], ignore_index=True)
    with pytest.raises(ValueError, match="exactly 1 row"):
        holder.predict(frame)


def test_unseen_category_does_not_raise(tiny_model):
    holder = ModelHolder()
    holder.load(tiny_model, summary_path=None)
    # A category never seen in training must be tolerated (handle_unknown).
    result = holder.predict(_row(STATUS="BRAND NEW STATUS", Group="never seen"))
    assert isinstance(result, float)


def test_summary_is_read_when_present(tiny_model, tmp_path):
    summary = tmp_path / "run_summary.json"
    summary.write_text(
        '{"model_type": "RandomForestRegressor", "data_version": "v2"}',
        encoding="utf-8",
    )
    holder = ModelHolder()
    holder.load(tiny_model, summary_path=summary)
    assert holder.model_type == "RandomForestRegressor"
    assert holder.data_version == "v2"


def test_missing_summary_keeps_loader_usable(tiny_model, tmp_path):
    holder = ModelHolder()
    holder.load(tiny_model, summary_path=tmp_path / "absent.json")
    assert holder.is_loaded is True
    assert holder.data_version is None


def test_reload_swaps_the_artifact(tiny_model, tmp_path):
    holder = ModelHolder()
    holder.load(tiny_model, summary_path=None)
    first = holder.predict(_row())

    other = tmp_path / "other.joblib"
    other.write_bytes(tiny_model.read_bytes())
    holder.reload(other, summary_path=None)

    assert holder.is_loaded is True
    assert holder.predict(_row()) == pytest.approx(first)
