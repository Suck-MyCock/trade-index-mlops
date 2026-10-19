"""Boundary-case tests for the request schemas.

Group 1 of 3. Every case here targets a constraint declared in
``app/schemas.py``; if a constraint is removed, one of these tests fails.
"""

from __future__ import annotations

import pytest
from app.schemas import MAX_YEAR, MIN_YEAR, UNKNOWN_CATEGORY, PredictRequest
from pydantic import ValidationError

VALID = {
    "year": 2024,
    "month": 6,
    "STATUS": "FINAL",
    "Subject": "Overseas Trade Indexes - Prices - OTP",
    "Group": "Export price indexes",
    "Series_title_1": "Price",
}


def test_valid_payload_is_accepted():
    request = PredictRequest(**VALID)
    assert request.year == 2024
    assert request.month == 6


@pytest.mark.parametrize("year", [MIN_YEAR - 1, MIN_YEAR - 100])
def test_year_below_minimum_is_rejected(year):
    with pytest.raises(ValidationError):
        PredictRequest(**{**VALID, "year": year})


@pytest.mark.parametrize("year", [MAX_YEAR + 1, 9999])
def test_year_above_maximum_is_rejected(year):
    with pytest.raises(ValidationError):
        PredictRequest(**{**VALID, "year": year})


@pytest.mark.parametrize("month", [0, -1, 13, 99])
def test_month_outside_1_to_12_is_rejected(month):
    with pytest.raises(ValidationError):
        PredictRequest(**{**VALID, "month": month})


@pytest.mark.parametrize("month", [1, 3, 6, 9, 12])
def test_all_quarter_months_are_accepted(month):
    assert PredictRequest(**{**VALID, "month": month}).month == month


@pytest.mark.parametrize("field", ["STATUS", "Subject", "Group", "Series_title_1"])
def test_blank_required_category_is_rejected(field):
    with pytest.raises(ValidationError):
        PredictRequest(**{**VALID, field: "   "})


@pytest.mark.parametrize("field", ["STATUS", "Subject", "Group", "Series_title_1"])
def test_missing_required_field_is_rejected(field):
    payload = {k: v for k, v in VALID.items() if k != field}
    with pytest.raises(ValidationError):
        PredictRequest(**payload)


def test_unknown_extra_field_is_rejected():
    with pytest.raises(ValidationError):
        PredictRequest(**VALID, unexpected="value")


@pytest.mark.parametrize("bad_year", ["2024", None, 2024.5, [2024]])
def test_non_integer_year_is_rejected(bad_year):
    with pytest.raises(ValidationError):
        PredictRequest(**{**VALID, "year": bad_year})


def test_blank_optional_categories_become_unknown():
    request = PredictRequest(**VALID, Series_title_2="", Series_title_3="   ")
    frame = request.to_frame()
    assert frame.loc[0, "Series_title_2"] == UNKNOWN_CATEGORY
    assert frame.loc[0, "Series_title_3"] == UNKNOWN_CATEGORY


def test_provided_optional_categories_are_kept():
    request = PredictRequest(**VALID, Series_title_2="BEEF AND VEAL")
    assert request.to_frame().loc[0, "Series_title_2"] == "BEEF AND VEAL"


def test_whitespace_is_stripped():
    request = PredictRequest(**{**VALID, "STATUS": "  FINAL  "})
    assert request.STATUS == "FINAL"


def test_to_frame_has_exactly_one_row():
    assert len(PredictRequest(**VALID).to_frame()) == 1


def test_to_frame_column_order_is_stable():
    expected = [
        "year",
        "month",
        "STATUS",
        "Subject",
        "Group",
        "Series_title_1",
        "Series_title_2",
        "Series_title_3",
        "Series_title_4",
    ]
    assert list(PredictRequest(**VALID).to_frame().columns) == expected
