"""Inference pipeline used by the serving layer.

Loads the trained serving pipeline once and exposes a simple ``predict`` over a
dict of raw feature values (same columns as the training data minus target).
"""

from functools import lru_cache

import pandas as pd

from loan_default.config.configuration import ConfigurationManager
from loan_default.utils.common import load_object


@lru_cache(maxsize=1)
def _load_model():
    # NOTE: this lru_cache holds the model for the process lifetime. An
    # out-of-band retrain that rewrites model.joblib will NOT be picked up until
    # the cache is cleared; the API's /train endpoint calls
    # ``_load_model.cache_clear()`` to do exactly that after retraining.
    model_file = ConfigurationManager().get_model_trainer_config().model_file
    return load_object(model_file)


@lru_cache(maxsize=1)
def _default_threshold() -> float:
    # Keep serving decisions consistent with the threshold used at evaluation.
    return ConfigurationManager().get_model_evaluation_config().decision_threshold


@lru_cache(maxsize=1)
def _feature_columns() -> tuple:
    cfg = ConfigurationManager().get_data_transformation_config()
    return tuple(cfg.numeric_columns) + tuple(cfg.categorical_columns)


class PredictionPipeline:
    def __init__(self):
        self.model = _load_model()
        self.threshold = _default_threshold()
        self.feature_columns = _feature_columns()

    def predict(self, features: dict, threshold: float | None = None) -> dict:
        if threshold is None:
            threshold = self.threshold
        missing = [c for c in self.feature_columns if c not in features]
        if missing:
            raise ValueError(f"Missing required feature field(s): {missing}")
        # Reindex to the training feature order so the preprocessor sees exactly
        # the columns it was fitted on (extra keys are ignored).
        df = pd.DataFrame([features]).reindex(columns=list(self.feature_columns))
        proba = float(self.model.predict_proba(df)[:, 1][0])
        is_default = proba >= threshold
        return {
            "default_probability": round(proba, 4),
            "prediction": "defaulter" if is_default else "non-defaulter",
            "is_defaulter": bool(is_default),
        }
