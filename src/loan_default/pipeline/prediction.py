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
    model_file = ConfigurationManager().get_model_trainer_config().model_file
    return load_object(model_file)


@lru_cache(maxsize=1)
def _default_threshold() -> float:
    # Keep serving decisions consistent with the threshold used at evaluation.
    return ConfigurationManager().get_model_evaluation_config().decision_threshold


class PredictionPipeline:
    def __init__(self):
        self.model = _load_model()
        self.threshold = _default_threshold()

    def predict(self, features: dict, threshold: float | None = None) -> dict:
        if threshold is None:
            threshold = self.threshold
        df = pd.DataFrame([features])
        proba = float(self.model.predict_proba(df)[:, 1][0])
        is_default = proba >= threshold
        return {
            "default_probability": round(proba, 4),
            "prediction": "defaulter" if is_default else "non-defaulter",
            "is_defaulter": bool(is_default),
        }
