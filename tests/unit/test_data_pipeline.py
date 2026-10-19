"""Boundary-case tests for the v1 to v2 data pipeline.

Every cleaning rule in ``ml/data_pipeline.py`` is exercised here with small
synthetic frames, so a change to the transformation logic fails a test rather
than silently altering what the model is trained on.

These tests deliberately avoid the real 23 MB CSV: the units of behaviour are
the rules, not the file, and a synthetic frame makes each edge case explicit.
"""

from __future__ import annotations

import pandas as pd
import pytest
from ml.data_pipeline import (
    DROP_COLUMNS,
    LEAKY_COLUMNS,
    TARGET,
    UNKNOWN,
    add_time_features,
    clean,
    drop_constant_columns,
    load_raw,
    normalize_units,
    select_target_units,
)

RAW_COLUMNS = [
    "Series_reference",
    "Period",
    "Data_value",
    "STATUS",
    "UNITS",
    "MAGNTUDE",
    "Subject",
    "Group",
    "Series_title_1",
    "Series_title_2",
    "Series_title_3",
    "Series_title_4",
    "Series_title_5",
]


def make_row(**overrides) -> dict:
    """One well-formed raw row; override individual fields per test."""
    row = {
        "Series_reference": "OTVA.SEA2AA1",
        "Period": "2020.06",
        "Data_value": "369.1357763",
        "STATUS": "FINAL",
        "UNITS": "Index",
        "MAGNTUDE": "0",
        "Subject": "Overseas Trade Indexes - Prices - OTP",
        "Group": "Export price indexes",
        "Series_title_1": "Price",
        "Series_title_2": "BEEF AND VEAL",
        "Series_title_3": "June 2002 Quarter Base (=1000)",
        "Series_title_4": "Trend",
        "Series_title_5": "",
    }
    row.update(overrides)
    return row


def make_frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows) if rows else [make_row()], columns=RAW_COLUMNS)


# ---------------------------------------------------------------- normalize
def test_normalize_units_lowercases_index():
    """The source mixes 'Index' and 'index'; both must become one category."""
    result = normalize_units(pd.Series(["Index", "index", "INDEX"]))
    assert set(result) == {"Index"}


def test_normalize_units_leaves_other_units_alone():
    result = normalize_units(pd.Series(["Dollars", "Percent"]))
    assert list(result) == ["Dollars", "Percent"]


def test_normalize_units_strips_whitespace():
    result = normalize_units(pd.Series(["  index  "]))
    assert result.iloc[0] == "Index"


# -------------------------------------------------------------------- clean
def test_clean_renames_misspelled_column():
    out = clean(make_frame())
    assert "MAGNITUDE" in out.columns
    assert "MAGNTUDE" not in out.columns


def test_clean_drops_leaky_and_empty_columns():
    out = clean(make_frame())
    for column in LEAKY_COLUMNS + DROP_COLUMNS:
        assert column not in out.columns


def test_clean_coerces_target_to_numeric():
    out = clean(make_frame())
    assert pd.api.types.is_float_dtype(out[TARGET])


def test_clean_drops_rows_without_a_target():
    frame = make_frame(
        make_row(Period="2020.03"),
        make_row(Period="2020.06", Data_value=""),
        make_row(Period="2020.09"),
    )
    out = clean(frame)
    assert len(out) == 2


def test_clean_drops_rows_with_unparseable_target():
    frame = make_frame(
        make_row(Period="2020.03"),
        make_row(Period="2020.06", Data_value="not-a-number"),
    )
    assert len(clean(frame)) == 1


def test_clean_fills_blank_categories_with_unknown():
    frame = make_frame(make_row(Series_title_3=""))
    out = clean(frame)
    assert out["Series_title_3"].iloc[0] == UNKNOWN


def test_clean_fills_nan_categories_with_unknown():
    frame = make_frame()
    frame.loc[0, "Series_title_2"] = None
    out = clean(frame)
    assert out["Series_title_2"].iloc[0] == UNKNOWN


def test_clean_is_idempotent_on_already_clean_data():
    """Running clean twice must not change the frame; the API relies on this."""
    once = clean(make_frame())
    twice = clean(once)
    pd.testing.assert_frame_equal(
        once.sort_index(axis=1), twice.sort_index(axis=1), check_dtype=False
    )


# -------------------------------------------------------- select_target_units
def test_select_target_units_keeps_only_index_rows():
    frame = clean(
        make_frame(
            make_row(Period="2020.03", UNITS="Index"),
            make_row(Period="2020.06", UNITS="Dollars"),
            make_row(Period="2020.09", UNITS="Percent"),
        )
    )
    out = select_target_units(frame)
    assert len(out) == 1


def test_select_target_units_drops_the_units_column():
    out = select_target_units(clean(make_frame()))
    assert "UNITS" not in out.columns


def test_select_target_units_can_target_a_different_unit():
    frame = clean(make_frame(make_row(UNITS="Dollars", Data_value="1000")))
    assert len(select_target_units(frame, units="Dollars")) == 1


def test_select_target_units_on_empty_result():
    frame = clean(make_frame(make_row(UNITS="Percent", Data_value="1.5")))
    out = select_target_units(frame, units="Index")
    assert len(out) == 0
    assert "UNITS" not in out.columns


# ----------------------------------------------------- drop_constant_columns
def test_drop_constant_columns_removes_single_value_column():
    frame = pd.DataFrame({"keep": [1, 2, 3], "constant": ["x", "x", "x"]})
    out, kept = drop_constant_columns(frame, features=["keep", "constant"])
    assert "constant" not in out.columns
    assert kept == ["keep"]


def test_drop_constant_columns_keeps_everything_when_all_vary():
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out, kept = drop_constant_columns(frame, features=["a", "b"])
    assert kept == ["a", "b"]
    assert list(out.columns) == ["a", "b"]


def test_drop_constant_columns_handles_empty_frame():
    frame = pd.DataFrame({"a": pd.Series([], dtype=float)})
    out, kept = drop_constant_columns(frame, features=["a"])
    assert "a" not in out.columns
    assert kept == []


# ------------------------------------------------------- add_time_features
def test_add_time_features_parses_year_and_month():
    out = add_time_features(pd.DataFrame({"Period": ["1971.06"]}))
    assert out["year"].iloc[0] == 1971
    assert out["month"].iloc[0] == 6


@pytest.mark.parametrize(
    "period,expected_year,expected_month",
    [
        ("1925.03", 1925, 3),
        ("2024.12", 2024, 12),
        ("2000.09", 2000, 9),
    ],
)
def test_add_time_features_handles_quarter_end_months(period, expected_year, expected_month):
    out = add_time_features(pd.DataFrame({"Period": [period]}))
    assert out["year"].iloc[0] == expected_year
    assert out["month"].iloc[0] == expected_month


@pytest.mark.parametrize("bad", ["", "not-a-period", "2020", "2020-06", "20.6"])
def test_add_time_features_returns_nan_for_unparseable_period(bad):
    """A malformed period must become NaN, never raise."""
    out = add_time_features(pd.DataFrame({"Period": [bad]}))
    assert pd.isna(out["year"].iloc[0])
    assert pd.isna(out["month"].iloc[0])


def test_add_time_features_does_not_mutate_input():
    frame = pd.DataFrame({"Period": ["2020.06"]})
    add_time_features(frame)
    assert list(frame.columns) == ["Period"]


# ----------------------------------------------------------------- load_raw
def test_load_raw_keeps_everything_as_text(tmp_path):
    """load_raw must not let pandas guess types; the pipeline converts on purpose."""
    path = tmp_path / "raw.csv"
    path.write_text("Period,Data_value,STATUS\n2020.06,369.1,FINAL\n", encoding="utf-8")
    out = load_raw(path)
    assert out["Period"].iloc[0] == "2020.06"
    assert out["Data_value"].iloc[0] == "369.1"


def test_load_raw_treats_blank_cells_as_missing(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("A,B\n1,\n", encoding="utf-8")
    out = load_raw(path)
    assert out["B"].isna().all()


def test_load_raw_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "absent.csv")
