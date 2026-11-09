"""Stage 3 - Data transformation.

Builds a scikit-learn ColumnTransformer (impute + scale numerics, impute +
one-hot encode categoricals), splits the data stratified on the target, fits
the preprocessor on the training split only, and persists both the fitted
preprocessor and the raw train/test splits for the downstream stages.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from loan_default.entity.config_entity import DataTransformationConfig
from loan_default.logger import logger
from loan_default.utils.common import save_object


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, self.config.numeric_columns),
                ("cat", categorical_pipe, self.config.categorical_columns),
            ]
        )

    def transform(self) -> tuple[str, str, str]:
        df = pd.read_csv(self.config.data_file)
        target = self.config.target_column

        # Guard: never fabricate labels. Validation rejects missing targets, but
        # drop any that slip through here rather than encoding them as class 0.
        n_missing = int(df[target].isna().sum())
        if n_missing:
            logger.warning("Dropping %d row(s) with a missing target.", n_missing)
            df = df.dropna(subset=[target])

        # Encode target to 0/1 (yes -> 1 = defaulter). Assert the label set is
        # {yes, no} first: the "== 'yes'" rule maps anything else to 0, so an
        # unexpected label would be silently mislabeled. This mirrors the
        # validation gate and defends against direct invocation of this stage.
        labels = set(df[target].astype(str).str.strip().str.lower().unique())
        if not labels.issubset({"yes", "no"}):
            raise ValueError(
                f"Target '{target}' has unexpected labels {sorted(labels)}; "
                "expected only 'yes'/'no' (case-insensitive)."
            )
        df[target] = (df[target].astype(str).str.strip().str.lower() == "yes").astype(int)

        train_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df[target],
        )

        preprocessor = self._build_preprocessor()
        feature_cols = self.config.numeric_columns + self.config.categorical_columns
        preprocessor.fit(train_df[feature_cols])

        save_object(self.config.preprocessor_file, preprocessor)
        train_df.to_csv(self.config.train_file, index=False)
        test_df.to_csv(self.config.test_file, index=False)

        logger.info(
            "Transformation done: train=%d test=%d; preprocessor -> %s",
            len(train_df),
            len(test_df),
            self.config.preprocessor_file,
        )
        return (
            str(self.config.train_file),
            str(self.config.test_file),
            str(self.config.preprocessor_file),
        )
