"""FastAPI service exposing the trade-index model.

Two endpoints:

    GET  /health    liveness + whether the model artifact is loaded
    POST /predict   score one observation

The model is loaded once during startup (``lifespan``) rather than per request,
so a cold start pays the deserialisation cost exactly once.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from app.model_loader import ModelNotLoadedError, model_holder
from app.schemas import HealthResponse, PredictRequest, PredictResponse

# Deployment environment. CD pipelines write this into the .env they generate
# from GitHub Environments, so a running container can say which environment it
# was built for.
APP_ENV = os.environ.get("APP_ENV", "local")

# Uvicorn configures its own loggers but leaves application loggers without a
# handler, so ``logger.info`` here would be silently dropped and the "model
# loaded" line would never reach `docker logs`. Attach one StreamHandler to
# this module's logger so startup diagnostics are always visible.
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load the model before the app starts accepting traffic.

    A missing artifact is logged but does not crash the process: the service
    still answers ``/health`` with ``model_loaded: false``, which is far easier
    to diagnose from a container log than a silent restart loop.
    """
    try:
        model_holder.load()
        logger.info(
            "model loaded: type=%s data_version=%s",
            model_holder.model_type,
            model_holder.data_version,
        )
    except FileNotFoundError as exc:
        logger.error("model artifact missing: %s", exc)
    yield


app = FastAPI(
    title="Trade Index Predictor",
    description="Scores NZ overseas trade index values from period and series metadata.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Report whether the service can serve predictions."""
    return HealthResponse(
        status="ok",
        environment=APP_ENV,
        model_loaded=model_holder.is_loaded,
        model_type=model_holder.model_type,
        data_version=model_holder.data_version,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["predict"],
)
def predict(request: PredictRequest) -> PredictResponse:
    """Score one trade-index observation.

    Returns 503 when the model artifact is unavailable, so a caller can tell
    "the service is up but not ready" apart from "your request was invalid"
    (which the framework reports as 422).
    """
    try:
        prediction = model_holder.predict(request.to_frame())
    except ModelNotLoadedError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model is not loaded",
        ) from exc

    return PredictResponse(
        prediction=prediction,
        model_type=model_holder.model_type or "unknown",
        data_version=model_holder.data_version or "unknown",
    )
