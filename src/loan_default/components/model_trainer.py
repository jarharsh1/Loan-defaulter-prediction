"""Stage 4 - Model training.

Loads the fitted preprocessor, transforms the training split, fits a
RandomForest (a stronger successor to the original decision tree), and saves a
single serving artifact: an sklearn Pipeline bundling preprocessor + model, so
the API only has to load one object and can accept raw feature rows.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from loan_default.entity.config_entity import ModelTrainerConfig
from loan_default.logger import logger
from loan_default.utils.common import load_object, save_object


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self) -> str:
        train_df = pd.read_csv(self.config.train_file)
        target = self.config.target_column
        X = train_df.drop(columns=[target])
        y = train_df[target]

        preprocessor = load_object(self.config.preprocessor_file)

        model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

        # Bundle preprocessing + model so serving loads one artifact.
        # NOTE: this .fit() refits the preprocessor on the training rows (the
        # same train-only data it was already fitted on in transformation), so
        # the served preprocessor has no test leakage. The standalone
        # preprocessor.joblib from the transformation stage is therefore not used
        # at serving time; it is kept only for lineage / inspection.
        serving_pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model)]
        )
        serving_pipeline.fit(X, y)

        save_object(self.config.model_file, serving_pipeline)
        logger.info(
            "Trained RandomForest(n_estimators=%d, max_depth=%s) -> %s",
            self.config.n_estimators,
            self.config.max_depth,
            self.config.model_file,
        )
        return str(self.config.model_file)
