"""Request and response models for the trade-index prediction API.

All input validation lives here, which is what makes the boundary-case tests
possible: every constraint declared with ``Field(...)`` is enforced before the
request reaches the model.

The field names mirror the v2 dataset's column names exactly (``STATUS``,
``Series_title_1``, ...), so a validated request can be turned into a DataFrame
without any renaming step that could silently drift from training.
"""

from __future__ import annotations

from typing import Annotated

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# Reasonable bounds for the trade-index period. The dataset covers 1925-2024;
# the wider window leaves room for future quarters without accepting nonsense.
MIN_YEAR = 1900
MAX_YEAR = 2100

# Filler used when a categorical field is left blank. Must match the value the
# data pipeline writes for missing categories, otherwise training and serving
# would encode the same thing differently.
UNKNOWN_CATEGORY = "Unknown"

ShortText = Annotated[str, Field(min_length=1, max_length=200)]


class PredictRequest(BaseModel):
    """One trade-index observation to score."""

    # extra="forbid" turns a typo in a client's JSON into a 422 instead of a
    # silently ignored field. str_strip_whitespace keeps " 2001 " usable.
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    year: int = Field(strict=True, ge=MIN_YEAR, le=MAX_YEAR)
    month: int = Field(strict=True, ge=1, le=12)

    STATUS: ShortText
    Subject: ShortText
    Group: ShortText
    Series_title_1: ShortText

    # These three are genuinely sparse in the source data, so blank is allowed
    # and is normalised to "Unknown" before reaching the model.
    Series_title_2: str = ""
    Series_title_3: str = ""
    Series_title_4: str = ""

    def to_frame(self) -> pd.DataFrame:
        """Build a one-row DataFrame in the exact column order used in training.

        Column order matters because the saved ``ColumnTransformer`` selects
        columns by name; getting it wrong would raise at predict time rather
        than produce a quiet wrong answer, but keeping it deterministic makes
        the tests meaningful.
        """
        return pd.DataFrame(
            [
                {
                    "year": self.year,
                    "month": self.month,
                    "STATUS": self.STATUS,
                    "Subject": self.Subject,
                    "Group": self.Group,
                    "Series_title_1": self.Series_title_1,
                    "Series_title_2": self._or_unknown(self.Series_title_2),
                    "Series_title_3": self._or_unknown(self.Series_title_3),
                    "Series_title_4": self._or_unknown(self.Series_title_4),
                }
            ]
        )

    @staticmethod
    def _or_unknown(value: str) -> str:
        cleaned = value.strip()
        return cleaned if cleaned else UNKNOWN_CATEGORY


class PredictResponse(BaseModel):
    """The scored observation plus enough provenance to interpret it."""

    prediction: float
    model_type: str
    data_version: str


class HealthResponse(BaseModel):
    """Liveness/readiness payload.

    ``environment`` comes from the ``APP_ENV`` variable that the CD pipeline
    writes into the generated ``.env``, so a running container can be traced
    back to the environment it was built for.
    """

    status: str
    environment: str
    model_loaded: bool
    model_type: str | None = None
    data_version: str | None = None
