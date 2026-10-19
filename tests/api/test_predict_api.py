"""Boundary-case tests for the HTTP surface.

Group 3 of 3. These drive the real FastAPI app through ``TestClient`` so the
whole request path — validation, routing, model call, response model — is
covered end to end.
"""

from __future__ import annotations

import pytest
from app.main import app
from app.model_loader import model_holder
from fastapi.testclient import TestClient

VALID_PAYLOAD = {
    "year": 2024,
    "month": 6,
    "STATUS": "FINAL",
    "Subject": "Overseas Trade Indexes - Prices - OTP",
    "Group": "Export price indexes",
    "Series_title_1": "Price",
    "Series_title_2": "BEEF AND VEAL",
    "Series_title_3": "June 2002 Quarter Base (=1000)",
    "Series_title_4": "Trend",
}


@pytest.fixture(scope="module")
def client():
    """Start the app once for the module, running the real lifespan hook."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_type"]
    assert body["data_version"] == "v2"


def test_health_reports_environment(client):
    """APP_ENV drives the reported environment; it defaults to "local"."""
    body = client.get("/health").json()
    assert "environment" in body
    assert isinstance(body["environment"], str)
    assert body["environment"]


def test_predict_valid_payload_returns_prediction(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["prediction"], float)
    assert body["data_version"] == "v2"


def test_predict_is_deterministic(client):
    first = client.post("/predict", json=VALID_PAYLOAD).json()["prediction"]
    second = client.post("/predict", json=VALID_PAYLOAD).json()["prediction"]
    assert first == second


@pytest.mark.parametrize(
    "mutation,description",
    [
        ({"year": 1800}, "year below bound"),
        ({"year": 2200}, "year above bound"),
        ({"month": 0}, "month zero"),
        ({"month": 13}, "month above 12"),
        ({"STATUS": ""}, "blank required category"),
        ({"year": "2024"}, "year as string"),
    ],
)
def test_predict_invalid_payload_returns_422(client, mutation, description):
    payload = {**VALID_PAYLOAD, **mutation}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, f"{description} should be rejected"


REQUIRED_FIELDS = ["year", "month", "STATUS", "Subject", "Group", "Series_title_1"]
OPTIONAL_FIELDS = ["Series_title_2", "Series_title_3", "Series_title_4"]


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_predict_missing_required_field_returns_422(client, field):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != field}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize("field", OPTIONAL_FIELDS)
def test_predict_missing_optional_field_still_scores(client, field):
    """The three sparse series titles default to an empty string on purpose.

    Omitting them must succeed and be encoded as "Unknown", otherwise a client
    that genuinely has no value for them could never call the API.
    """
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != field}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_predict_extra_field_returns_422(client):
    payload = {**VALID_PAYLOAD, "colour": "red"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_empty_body_returns_422(client):
    assert client.post("/predict", json={}).status_code == 422


def test_predict_unseen_categories_still_scores(client):
    payload = {
        **VALID_PAYLOAD,
        "STATUS": "NEVER SEEN",
        "Subject": "NEVER SEEN EITHER",
        "Group": "not a real group",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert isinstance(response.json()["prediction"], float)


def test_predict_without_model_returns_503(client):
    """Simulate a container that started before the artifact existed."""
    saved_pipeline = model_holder._pipeline  # noqa: SLF001 - deliberate test hook
    saved_type = model_holder._model_type  # noqa: SLF001
    try:
        model_holder._pipeline = None  # noqa: SLF001
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 503
    finally:
        model_holder._pipeline = saved_pipeline  # noqa: SLF001
        model_holder._model_type = saved_type  # noqa: SLF001


def test_health_reflects_missing_model(client):
    saved_pipeline = model_holder._pipeline  # noqa: SLF001
    try:
        model_holder._pipeline = None  # noqa: SLF001
        body = client.get("/health").json()
        assert body["model_loaded"] is False
    finally:
        model_holder._pipeline = saved_pipeline  # noqa: SLF001


def test_openapi_schema_is_served(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]
