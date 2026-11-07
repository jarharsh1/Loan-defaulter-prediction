"""Unit tests for the schema-based data validation component."""

import pandas as pd

from loan_default.components.data_validation import DataValidation
from loan_default.entity.config_entity import DataValidationConfig
from loan_default.utils.common import load_json, read_yaml

SCHEMA = read_yaml("schema.yaml")


def _config(tmp_path, data_file):
    return DataValidationConfig(
        root_dir=tmp_path,
        data_file=data_file,
        status_file=tmp_path / "status.json",
        schema=SCHEMA,
    )


def _valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "checking_balance": ["< 0 DM"],
            "months_loan_duration": [12],
            "credit_history": ["good"],
            "purpose": ["education"],
            "amount": [2000],
            "savings_balance": ["< 100 DM"],
            "employment_duration": ["1 - 4 years"],
            "percent_of_income": [2],
            "years_at_residence": [3],
            "age": [35],
            "other_credit": ["none"],
            "housing": ["own"],
            "existing_loans_count": [1],
            "job": ["skilled"],
            "dependents": [1],
            "phone": ["yes"],
            "default": ["no"],
        }
    )


def test_valid_data_passes(tmp_path):
    f = tmp_path / "ok.csv"
    _valid_frame().to_csv(f, index=False)
    assert DataValidation(_config(tmp_path, f)).validate() is True
    status = load_json(tmp_path / "status.json")
    assert status["validation_passed"] is True
    assert status["errors"] == []


def test_missing_column_fails(tmp_path):
    df = _valid_frame().drop(columns=["age"])
    f = tmp_path / "bad.csv"
    df.to_csv(f, index=False)
    assert DataValidation(_config(tmp_path, f)).validate() is False


def test_bad_target_labels_fail(tmp_path):
    df = _valid_frame()
    df["default"] = ["maybe"]
    f = tmp_path / "bad_target.csv"
    df.to_csv(f, index=False)
    assert DataValidation(_config(tmp_path, f)).validate() is False


def test_missing_target_fails(tmp_path):
    # A row with a valid label plus a row whose target is blank.
    good = _valid_frame()
    missing = _valid_frame()
    missing["default"] = [None]
    df = pd.concat([good, missing], ignore_index=True)
    f = tmp_path / "missing_target.csv"
    df.to_csv(f, index=False)
    assert DataValidation(_config(tmp_path, f)).validate() is False
