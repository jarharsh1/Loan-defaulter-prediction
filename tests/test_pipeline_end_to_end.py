"""End-to-end smoke test: run the whole pipeline, then score via the API.

Runs on the real (small) dataset so it stays fast but exercises every stage and
the serving layer together.
"""

import json

from fastapi.testclient import TestClient


def test_full_pipeline_and_api(tmp_path, monkeypatch):
    # Run the pipeline in-process (uses repo config, writes to ./artifacts).
    from main import main as run_pipeline

    run_pipeline()

    # Metrics file exists and is sane.
    with open("artifacts/model_evaluation/metrics.json") as f:
        metrics = json.load(f)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    # A trivially-beatable model would be worse than this on German credit.
    assert metrics["roc_auc"] > 0.6

    # Serve a prediction.
    from app.main import app

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200

    payload = {
        "checking_balance": "< 0 DM",
        "months_loan_duration": 24,
        "credit_history": "critical",
        "purpose": "furniture/appliances",
        "amount": 5000,
        "savings_balance": "< 100 DM",
        "employment_duration": "1 - 4 years",
        "percent_of_income": 4,
        "years_at_residence": 2,
        "age": 30,
        "other_credit": "none",
        "housing": "own",
        "existing_loans_count": 2,
        "job": "skilled",
        "dependents": 1,
        "phone": "no",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in {"defaulter", "non-defaulter"}
    assert 0.0 <= body["default_probability"] <= 1.0
