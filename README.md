# Loan Default Prediction — End-to-End MLOps

Predict whether a loan applicant is likely to **default**, built as a complete,
reproducible MLOps project around the classic German-credit dataset
(`credit.csv`, 1000 rows, 16 features).

This repository started life as a single exploratory notebook
(`Loan_Defaulter.ipynb`, a decision-tree walkthrough) and has been rebuilt into
a modular, config-driven, testable, servable pipeline.

## Architecture

```
            data/raw/credit.csv  (versioned reference data)
                     │
   ┌─────────────────▼──────────────────────────────────────────┐
   │                   Training pipeline (DVC DAG)                │
   │                                                              │
   │  1. Data Ingestion      copy → artifacts/data_ingestion      │
   │  2. Data Validation     schema.yaml contract → status.json   │
   │  3. Data Transformation ColumnTransformer + stratified split │
   │  4. Model Trainer       RandomForest in an sklearn Pipeline  │
   │  5. Model Evaluation    metrics.json + MLflow run            │
   └─────────────────┬──────────────────────────────────────────┘
                     │  model.joblib (preprocessor + model bundled)
                     ▼
            FastAPI serving  (/health, /predict, /train)  → Docker
```

Everything is driven by three files:

| File | Purpose |
|------|---------|
| `config/config.yaml` | Paths / artifact locations per stage |
| `params.yaml` | Tunable hyperparameters (tracked by DVC & MLflow) |
| `schema.yaml` | Data contract used by validation |

## Project layout

```
src/loan_default/
  components/     ingestion · validation · transformation · trainer · evaluation
  pipeline/       one runnable stage per component + prediction pipeline
  config/         ConfigurationManager (YAML → typed configs)
  entity/         frozen dataclass configs
  utils/          yaml/json/joblib helpers
  logger.py       rotating-file + stdout logging
  exception.py    error wrapper with file/line context
app/              FastAPI serving layer (schemas + endpoints)
tests/            unit (validation) + end-to-end (pipeline + API) tests
dvc.yaml          reproducible pipeline DAG
Dockerfile        trains at build, serves with uvicorn
.github/workflows CI: lint → train → test → build image
```

## Quickstart

```bash
# 1. Install (editable, with dev extras)
pip install -r requirements.txt

# 2. Run the full training pipeline
python main.py
#    → artifacts/model_trainer/model.joblib
#    → artifacts/model_evaluation/metrics.json

# 3. Serve the model
uvicorn app.main:app --reload
#    POST http://127.0.0.1:8000/predict   (see /docs for the schema)
```

### Reproduce with DVC

```bash
dvc repro          # re-runs only the stages whose deps/params changed
dvc metrics show   # prints the evaluation metrics
```

### Experiment tracking (MLflow)

Each evaluation logs params + metrics + the model to a local `mlruns/` store by
default (set `model_evaluation.mlflow_uri` in `config/config.yaml` to point at a
remote server):

```bash
mlflow ui          # browse runs at http://127.0.0.1:5000
```

### Run with Docker

```bash
docker build -t loan-default .
docker run -p 8000:8000 loan-default
```

### Tests

```bash
pytest -q
```

## Example prediction request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "phone": "no"
  }'
```

```json
{ "default_probability": 0.71, "prediction": "defaulter", "is_defaulter": true }
```

## Dataset note

`credit.csv` is the reference dataset used throughout. The pipeline is
data-agnostic: drop in a new file matching `schema.yaml`, run `dvc repro`, and
every downstream stage re-runs automatically.

## Use cases

This repository is really a **reusable MLOps template for any tabular binary- or
multi-class classification problem** — the credit-default dataset is just the
worked example. The ingestion → validation → transformation → training →
evaluation pipeline, DVC DAG, MLflow tracking, FastAPI serving, Docker and CI
stay the same; you change the data and the `schema.yaml` contract.

| Domain | Swap in | Predict |
|--------|---------|---------|
| **Credit / lending** *(shipped example)* | applicant records | loan default risk |
| **Healthcare** | patient features | readmission, no-show, or disease risk |
| **Churn** | customer/account features | will the customer leave |
| **Insurance** | policyholder features | claim likelihood / fraud |
| **HR** | employee features | attrition risk |
| **Marketing** | lead features | conversion likelihood |

To retarget: replace `data/raw/…`, update `schema.yaml` (columns, numeric vs.
categorical, target), tune `params.yaml`, and run `dvc repro`. Everything
downstream — the model artifact, metrics, and the `/predict` API — regenerates.
