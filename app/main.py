"""FastAPI serving layer for the loan-default model.

Endpoints:
  GET  /health   - liveness/readiness, reports whether the model is loaded
  POST /predict  - score a single loan application
  POST /train    - re-run the full training pipeline (ops convenience)
"""

from fastapi import FastAPI, HTTPException

from app.schemas import LoanApplication, PredictionResponse
from loan_default.logger import logger

app = FastAPI(
    title="Loan Default Prediction API",
    version="0.1.0",
    description="Predicts whether a loan applicant is likely to default.",
)

_pipeline = None


def _get_pipeline():
    """Lazy-load the model so the app can boot even before training runs."""
    global _pipeline
    if _pipeline is None:
        from loan_default.pipeline.prediction import PredictionPipeline

        _pipeline = PredictionPipeline()
    return _pipeline


@app.get("/health")
def health() -> dict:
    model_ready = True
    try:
        _get_pipeline()
    except Exception:  # model artifact not present yet
        model_ready = False
    return {"status": "ok", "model_loaded": model_ready}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication) -> PredictionResponse:
    try:
        pipeline = _get_pipeline()
    except Exception as exc:
        logger.error("Model not available: %s", exc)
        raise HTTPException(status_code=503, detail="Model not trained yet.") from exc

    result = pipeline.predict(application.model_dump())
    return PredictionResponse(**result)


@app.post("/train")
def train() -> dict:
    """Trigger a full retrain. Resets the cached model on success."""
    global _pipeline
    try:
        from loan_default.pipeline.prediction import _default_threshold, _load_model
        from main import main as run_pipeline

        run_pipeline()
        # Drop the caches so the next request picks up the freshly trained model
        # (and any changed threshold), not the stale lru_cache entries.
        _load_model.cache_clear()
        _default_threshold.cache_clear()
        _pipeline = None
        return {"status": "retrained"}
    except Exception as exc:
        logger.error("Retrain failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
