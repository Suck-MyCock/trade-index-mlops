"""Load the trained pipeline once and serve predictions from it.

Why a holder object instead of loading inside the request handler:

* Loading a RandomForest from disk takes hundreds of milliseconds; doing it per
  request would dominate latency.
* A single shared instance keeps memory flat.
* ``reload()`` exists so a test can point the API at a different artifact
  without restarting the process.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = Path("models/model.joblib")
DEFAULT_SUMMARY_PATH = Path("models/run_summary.json")


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is attempted before the model is available."""


def _estimator_name(pipeline) -> str:
    """Return the concrete estimator class name, not the wrapper's.

    ``type(pipeline).__name__`` is always ``"Pipeline"``, which tells a caller
    nothing. The useful name is the last step's class, so a health check can
    report ``RandomForestRegressor``.
    """
    steps = getattr(pipeline, "named_steps", None)
    if steps:
        last = list(steps.values())[-1]
        return type(last).__name__
    return type(pipeline).__name__


class ModelHolder:
    """Holds the fitted sklearn pipeline and its provenance metadata."""

    def __init__(self) -> None:
        self._pipeline = None
        self._model_type: str | None = None
        self._data_version: str | None = None

    # ------------------------------------------------------------------ state
    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def model_type(self) -> str | None:
        return self._model_type

    @property
    def data_version(self) -> str | None:
        return self._data_version

    # ---------------------------------------------------------------- loading
    def load(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        summary_path: str | Path | None = DEFAULT_SUMMARY_PATH,
    ) -> None:
        """Load the pipeline from disk.

        Raises ``FileNotFoundError`` with an actionable message rather than
        letting a bare joblib error escape: a missing artifact is an operator
        problem, not a server bug.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"model artifact not found at {path}. Run `python -m ml.train` to create it."
            )

        self._pipeline = joblib.load(path)
        self._model_type = _estimator_name(self._pipeline)
        self._data_version = None

        if summary_path is not None:
            summary = Path(summary_path)
            if summary.exists():
                meta = json.loads(summary.read_text(encoding="utf-8"))
                self._model_type = meta.get("model_type") or self._model_type
                self._data_version = meta.get("data_version")

    def reload(self, model_path: str | Path, summary_path: str | Path | None = None) -> None:
        """Drop the current pipeline and load a different artifact."""
        self._pipeline = None
        self._model_type = None
        self._data_version = None
        self.load(model_path, summary_path)

    # ------------------------------------------------------------- prediction
    def predict(self, frame: pd.DataFrame) -> float:
        """Score a single-row feature frame and return the prediction."""
        if self._pipeline is None:
            raise ModelNotLoadedError("model is not loaded")

        if len(frame) != 1:
            raise ValueError(f"expected exactly 1 row, got {len(frame)}")

        return float(self._pipeline.predict(frame)[0])

    # ------------------------------------------------------------- metadata
    def categories(self) -> dict[str, list[str]]:
        """Return the categorical values the model was trained on.

        Read from the fitted ``OneHotEncoder`` rather than from the dataset.
        That matters for deployment: the inference image deliberately excludes
        ``data/``, so there is no CSV to read. The encoder's ``categories_``
        is exactly the set the model can encode, which is what a form needs.
        """
        if self._pipeline is None:
            return {}

        pipeline = self._pipeline
        prep = getattr(pipeline, "named_steps", {}).get("prep")
        if prep is None:
            return {}

        try:
            encoder = prep.named_transformers_["cat"]
            columns = prep.transformers_[0][2]
            categories = encoder.categories_
        except (AttributeError, KeyError, IndexError):
            return {}

        return {
            str(column): sorted(str(value) for value in values)
            # strict: the encoder stores one category array per input column,
            # so a length mismatch means the pipeline was not built the way
            # this method assumes. Failing loudly beats silently truncating.
            for column, values in zip(columns, categories, strict=True)
        }


# Module-level singleton used by the API. Tests can call ``.reload()`` on it.
model_holder = ModelHolder()
