"""Typed configuration objects for each pipeline stage.

Dataclasses keep the contract between the config manager and the components
explicit, so components never touch raw YAML.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_file: Path
    local_data_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_file: Path
    status_file: Path
    schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_file: Path
    preprocessor_file: Path
    train_file: Path
    test_file: Path
    target_column: str
    numeric_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)
    test_size: float = 0.2
    random_state: int = 42


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_file: Path
    preprocessor_file: Path
    model_file: Path
    target_column: str
    n_estimators: int
    max_depth: int
    min_samples_leaf: int
    class_weight: str
    random_state: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_file: Path
    model_file: Path
    metrics_file: Path
    target_column: str
    decision_threshold: float
    mlflow_uri: str
    all_params: dict
