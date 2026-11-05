"""Stage 2 - Data validation.

Checks the ingested file against ``schema.yaml``: all expected columns present,
no unexpected columns, dtypes compatible, target present with valid labels, and
no fully-null columns. Writes a status.json the pipeline can gate on.
"""

import pandas as pd

from loan_default.entity.config_entity import DataValidationConfig
from loan_default.logger import logger
from loan_default.utils.common import save_json


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate(self) -> bool:
        df = pd.read_csv(self.config.data_file)
        schema_cols: dict = self.config.schema["columns"]
        target = self.config.schema["target_column"]

        errors: list[str] = []

        # 1. Column set matches schema exactly.
        expected = set(schema_cols)
        actual = set(df.columns)
        missing = expected - actual
        extra = actual - expected
        if missing:
            errors.append(f"Missing columns: {sorted(missing)}")
        if extra:
            errors.append(f"Unexpected columns: {sorted(extra)}")

        # 2. Dtype compatibility (numeric-vs-object family).
        for col, expected_dtype in schema_cols.items():
            if col not in df.columns:
                continue
            actual_dtype = str(df[col].dtype)
            exp_numeric = expected_dtype.startswith(("int", "float"))
            act_numeric = actual_dtype.startswith(("int", "float"))
            if exp_numeric != act_numeric:
                errors.append(
                    f"Column '{col}' dtype mismatch: expected {expected_dtype}, got {actual_dtype}"
                )

        # 3. No completely empty columns.
        for col in df.columns:
            if df[col].isna().all():
                errors.append(f"Column '{col}' is entirely null")

        # 4. Target present and binary-ish.
        if target in df.columns:
            labels = set(df[target].dropna().unique())
            if not labels.issubset({"yes", "no"}):
                errors.append(f"Target '{target}' has unexpected labels: {labels}")

        status = {
            "validation_passed": len(errors) == 0,
            "n_rows": len(df),
            "n_columns": int(df.shape[1]),
            "errors": errors,
        }
        save_json(self.config.status_file, status)

        if errors:
            logger.error("Data validation FAILED: %s", errors)
        else:
            logger.info("Data validation passed (%d rows).", len(df))
        return status["validation_passed"]
