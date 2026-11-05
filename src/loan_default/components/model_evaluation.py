"""Stage 5 - Model evaluation.

Scores the serving pipeline on the held-out test split, writes a metrics.json,
and logs params + metrics + the model to MLflow. MLflow falls back to a local
``mlruns/`` directory when no tracking URI is configured, so this runs the same
locally and in CI without extra infrastructure.
"""

import pandas as pd

try:
    import mlflow
    import mlflow.sklearn

    _MLFLOW_AVAILABLE = True
except ImportError:  # MLflow is optional; evaluation still writes metrics.json
    _MLFLOW_AVAILABLE = False
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from loan_default.entity.config_entity import ModelEvaluationConfig
from loan_default.logger import logger
from loan_default.utils.common import load_object, save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self) -> dict:
        test_df = pd.read_csv(self.config.test_file)
        target = self.config.target_column
        X = test_df.drop(columns=[target])
        y = test_df[target]

        model = load_object(self.config.model_file)

        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= self.config.decision_threshold).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, proba)),
        }
        save_json(self.config.metrics_file, metrics)
        logger.info("Evaluation metrics: %s", metrics)

        self._log_to_mlflow(metrics)
        return metrics

    def _log_to_mlflow(self, metrics: dict) -> None:
        if not _MLFLOW_AVAILABLE:
            logger.warning("MLflow not installed; skipping experiment logging.")
            return
        if self.config.mlflow_uri:
            mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment("loan-default")

        with mlflow.start_run():
            for section, params in self.config.all_params.items():
                if isinstance(params, dict):
                    for k, v in params.items():
                        mlflow.log_param(f"{section}.{k}", v)
            mlflow.log_metrics(metrics)
            try:
                model = load_object(self.config.model_file)
                mlflow.sklearn.log_model(model, name="model")
            except Exception as exc:  # pragma: no cover - logging best-effort
                logger.warning("Could not log model to MLflow: %s", exc)
        logger.info("Logged run to MLflow experiment 'loan-default'.")
