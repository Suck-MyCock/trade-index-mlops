"""Data cleaning and feature engineering for the NZ trade index dataset.

This module is the single source of truth for how a raw trade-index record is
turned into model-ready features. The training script and the inference API
both call :func:`clean` and :func:`add_time_features`, which guarantees the
features seen at training time match the features seen at prediction time.

Raw dataset facts (verified against data/raw/raw_dataset.csv):
    * 104,613 rows, 13 columns
    * target ``Data_value`` has 901 missing values
    * ``MAGNTUDE`` is misspelled in the source data
    * ``Series_title_5`` is empty for all but 274 rows
    * ``Series_reference`` is a series identifier and would leak the target
    * ``Data_value`` mixes three incomparable scales:
      Dollars (mean 3.1e6, max 1.1e9), Index (mean 1.1e3, max 1.4e4) and
      Percent (mean 1.8). Training on the mixture lets the dollar rows
      dominate the loss, so v2 keeps Index rows only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# The column we predict.
TARGET = "Data_value"

# v2 keeps a single unit so the target has one comparable scale.
TARGET_UNITS = "Index"

# Identifier-like column: keeping it would let the model memorise each series
# instead of learning a general relationship, so it is dropped.
LEAKY_COLUMNS = ["Series_reference"]

# Columns that are almost entirely empty in the source data.
DROP_COLUMNS = ["Series_title_5"]

# Columns whose spelling is wrong in the source data.
RENAME_COLUMNS = {"MAGNTUDE": "MAGNITUDE"}

# Categorical features fed to the one-hot encoder.
CATEGORICAL_FEATURES = [
    "STATUS",
    "MAGNITUDE",
    "Subject",
    "Group",
    "Series_title_1",
    "Series_title_2",
    "Series_title_3",
    "Series_title_4",
]

# Numeric features derived from the Period column.
NUMERIC_FEATURES = ["year", "month"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns kept in the v2 CSV: feature inputs plus the target and the period key.
OUTPUT_COLUMNS = ["Period", TARGET] + FEATURE_COLUMNS

# Filled in when a categorical value is missing or blank.
UNKNOWN = "Unknown"


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV, keeping every column as text so we control conversions.

    ``dtype=str`` stops pandas from turning blank cells into NaN on its own and
    from guessing a numeric type for ``Period`` (which is ``YYYY.MM`` text).
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def normalize_units(series: pd.Series) -> pd.Series:
    """Make ``UNITS`` consistent: the source mixes ``Index`` and ``index``."""
    return series.str.strip().str.title()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the base cleaning rules and return a new frame.

    Steps:
        1. rename the misspelled ``MAGNTUDE`` column
        2. drop identifier and near-empty columns
        3. normalise the ``UNITS`` casing
        4. coerce ``Data_value`` to numeric and drop rows without a target
        5. fill remaining categorical gaps with ``Unknown``

    This function does not filter by unit; :func:`select_target_units` does
    that separately so the step is explicit and testable.
    """
    out = df.copy()

    out = out.rename(columns=RENAME_COLUMNS)

    out = out.drop(columns=[c for c in LEAKY_COLUMNS + DROP_COLUMNS if c in out.columns])

    if "UNITS" in out.columns:
        out["UNITS"] = normalize_units(out["UNITS"])

    # A row without a target is useless for supervised learning.
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    out = out.dropna(subset=[TARGET])

    for column in CATEGORICAL_FEATURES:
        if column in out.columns:
            out[column] = out[column].astype("string").fillna(UNKNOWN)
            out[column] = out[column].str.strip()
            out[column] = out[column].replace("", UNKNOWN)

    return out.reset_index(drop=True)


def select_target_units(df: pd.DataFrame, units: str = TARGET_UNITS) -> pd.DataFrame:
    """Keep only rows whose ``UNITS`` equals ``units`` and drop the column.

    After filtering, ``UNITS`` is constant, so it carries no information.
    """
    out = df[df["UNITS"] == units].copy()
    out = out.drop(columns=[c for c in ("UNITS",) if c in out.columns])
    return out.reset_index(drop=True)


def drop_constant_columns(
    df: pd.DataFrame, features: list[str] | None = None, min_unique: int = 2
) -> tuple[pd.DataFrame, list[str]]:
    """Drop feature columns that carry no signal (fewer than ``min_unique`` values).

    ``MAGNITUDE`` collapses to a single value once v2 filters to Index units, so
    it would only add a useless one-hot column. Returns the frame and the list
    of columns that survived, so callers can build the encoder from it.
    """
    features = list(features if features is not None else FEATURE_COLUMNS)
    kept = [c for c in features if c in df.columns and df[c].nunique() >= min_unique]
    dropped = [c for c in features if c in df.columns and c not in kept]

    if dropped:
        out = df.drop(columns=dropped)
    else:
        out = df.copy()
    return out, kept


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Split ``Period`` (``YYYY.MM``) into integer ``year`` and ``month``.

    Rows whose ``Period`` cannot be parsed get NaN and are dropped by the
    caller. The source data has no such rows, but the guard keeps the API from
    crashing on bad input.
    """
    out = df.copy()
    period = out["Period"].astype("string").str.strip()
    parts = period.str.extract(r"^(?P<year>\d{4})\.(?P<month>\d{2})$")
    out["year"] = pd.to_numeric(parts["year"], errors="coerce")
    out["month"] = pd.to_numeric(parts["month"], errors="coerce")
    return out


def build_v2_dataset(raw_path: str | Path, out_path: str | Path) -> pd.DataFrame:
    """Run the full v1 -> v2 pipeline and write the cleaned CSV.

    Returns the cleaned frame so callers (and tests) can inspect it directly.
    Constant feature columns are dropped automatically.
    """
    raw = load_raw(raw_path)
    cleaned = clean(raw)
    filtered = select_target_units(cleaned)
    featured = add_time_features(filtered)
    featured = featured.dropna(subset=NUMERIC_FEATURES + [TARGET])

    featured, _kept = drop_constant_columns(featured)

    ordered = [c for c in OUTPUT_COLUMNS if c in featured.columns]
    ordered += [c for c in featured.columns if c not in ordered]
    featured = featured[ordered]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(out_path, index=False)
    return featured


def main() -> None:
    """CLI entry point: ``python -m ml.data_pipeline``."""
    import argparse

    parser = argparse.ArgumentParser(description="Build the v2 trade-index dataset.")
    parser.add_argument("--input", default="data/raw/raw_dataset.csv")
    parser.add_argument("--output", default="data/processed/clean_v2.csv")
    args = parser.parse_args()

    frame = build_v2_dataset(args.input, args.output)
    print(f"Wrote {args.output}")
    print(f"rows={len(frame)}  columns={len(frame.columns)}")
    print(f"features={len(FEATURE_COLUMNS)}  target={TARGET}")


if __name__ == "__main__":
    main()
