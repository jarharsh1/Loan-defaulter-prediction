# Code Review — Loan Defaulter Prediction

Reviewer-facing deep-dive on the correctness findings in this repo (German-credit
tabular binary classification; RandomForest; DVC pipeline; MLflow; FastAPI
`/predict`) and the fixes that landed. Each finding gives **Where**, **Root
cause**, **Fix** (before/after from the actual commit), and the **regression
test** that locks it. Commits referenced: `d5e5114`, `270a46a`, `dbeeccb`.

The ML core (split-before-fit, train-only transforms, held-out metrics) was
audited and found correct — see "Verified correct, not changed" at the end.

---

## 1. Model artifact loaded twice during evaluation

**Where:** `src/loan_default/components/model_evaluation.py` — `evaluate()` and
`_log_to_mlflow()` (commit `d5e5114`).

**Root cause:** `evaluate()` loaded the serving pipeline with `load_object(...)`
to score the test set, and then `_log_to_mlflow()` called `load_object(...)`
again to log the same artifact to MLflow. The model was deserialized from disk
twice per run — wasted I/O, and a subtle correctness smell (two in-memory copies
that could diverge if the file changed between reads).

**Fix:** load once in `evaluate()` and pass the object into `_log_to_mlflow(model)`.

```python
# before
        self._log_to_mlflow(metrics)
        return metrics

    def _log_to_mlflow(self, metrics: dict) -> None:
        ...
            mlflow.log_metrics(metrics)
            try:
                model = load_object(self.config.model_file)   # second disk read
                mlflow.sklearn.log_model(model, name="model")

# after
        self._log_to_mlflow(metrics, model)
        return metrics

    def _log_to_mlflow(self, metrics: dict, model) -> None:
        ...
            # MLflow metrics must be numeric; drop any None (e.g. undefined AUC).
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
            try:
                mlflow.sklearn.log_model(model, name="model")
```

**Regression test:** covered indirectly by the end-to-end smoke test
(`tests/test_pipeline_end_to_end.py`), which runs evaluation and asserts
`metrics.json` is produced and sane. No behavioral change to metrics; this is a
pure de-duplication plus the None-filtering needed by finding #3.

---

## 2. `predict()` raised a bare `KeyError` (unhandled 500) on a missing feature

**Where:** `src/loan_default/pipeline/prediction.py` — `PredictionPipeline.predict()`
(commit `270a46a`).

**Root cause:** `predict()` built `pd.DataFrame([features])` directly from the
caller's dict. If a feature key was absent, the `ColumnTransformer` inside the
serving pipeline hit a `KeyError` deep in sklearn, surfacing as an opaque HTTP
500 with no indication of *which* field was missing. Column *order* was also at
the mercy of dict insertion order.

**Fix:** validate the expected feature columns up front (raise a clear
`ValueError` naming the missing fields), then `reindex` the one-row frame to the
exact training feature order so the preprocessor sees the columns it was fitted
on (extra keys ignored). Feature columns come from the transformation config
(`numeric + categorical`) via a new cached `_feature_columns()` helper.

```python
# before
    def predict(self, features: dict, threshold: float | None = None) -> dict:
        if threshold is None:
            threshold = self.threshold
        df = pd.DataFrame([features])
        proba = float(self.model.predict_proba(df)[:, 1][0])

# after
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
```

**Regression test:** `tests/test_prediction.py` —
`test_missing_feature_raises_clear_error` and `test_missing_feature_names_the_field`.
These bypass `__init__` (`PredictionPipeline.__new__`) and set only
`feature_columns`, so they assert the guard fires *before* any model is touched —
no trained artifact or full pipeline needed:

```python
def _pipeline_with_columns(columns):
    p = PredictionPipeline.__new__(PredictionPipeline)
    p.feature_columns = tuple(columns)
    p.threshold = 0.5
    p.model = None  # must not be reached when a field is missing
    return p

def test_missing_feature_raises_clear_error():
    p = _pipeline_with_columns(["amount", "age"])
    with pytest.raises(ValueError, match="Missing required feature field"):
        p.predict({"amount": 1000})  # 'age' missing
```

---

## 3. Single-class test split crashed ROC-AUC

**Where:** `src/loan_default/components/model_evaluation.py` — `evaluate()`
(commit `d5e5114`).

**Root cause:** `roc_auc_score(y, proba)` raises `ValueError` when `y` contains
only one class. On a tiny or degenerate test split this would crash the whole
evaluation stage (and the DVC run) instead of degrading gracefully.

**Fix:** guard on `y.nunique() < 2` and record `roc_auc: null` instead of
crashing. Because `metrics.json` can now hold a `None`, `_log_to_mlflow` filters
`None` values before `mlflow.log_metrics` (MLflow requires numeric metrics).

```python
# before
            "f1": float(f1_score(y, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, proba)),
        }

# after
            "f1": float(f1_score(y, preds, zero_division=0)),
        }
        # ROC-AUC needs both classes present; record null (not a crash) if the
        # test labels are single-class.
        if y.nunique() < 2:
            logger.warning("Only one class in y_true; ROC-AUC is undefined.")
            metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = float(roc_auc_score(y, proba))
```

**Regression test:** the end-to-end test asserts `0.0 <= metrics["roc_auc"] <= 1.0`
on the real (two-class) data, so the happy path is locked; the guard itself is a
defensive branch for degenerate splits. `None` is a deliberate, documented
sentinel meaning "not computable".

---

## 4. Target encoding silently mapped unknown labels to class 0

**Where:** `src/loan_default/components/data_transformation.py` — `transform()`
(commit `dbeeccb`).

**Root cause:** the target was encoded with
`(df[target].astype(str).str.lower() == "yes").astype(int)`. This maps `"yes"` →
1 and **everything else** → 0. A dirty or unexpected label (`"maybe"`, `"1"`, a
typo) would be silently folded into the negative class rather than failing. The
validation gate already checks labels ⊆ {yes, no}, but running the transformation
stage directly (or after a schema drift) bypassed that protection.

**Fix:** assert the label set is a subset of {yes, no} (case-insensitive,
whitespace-stripped) *before* encoding, and raise a clear error otherwise —
mirroring the validation gate so direct invocation is also defended.

```python
# before
        # Encode target to 0/1 (yes -> 1 = defaulter).
        df[target] = (df[target].astype(str).str.lower() == "yes").astype(int)

# after
        labels = set(df[target].astype(str).str.strip().str.lower().unique())
        if not labels.issubset({"yes", "no"}):
            raise ValueError(
                f"Target '{target}' has unexpected labels {sorted(labels)}; "
                "expected only 'yes'/'no' (case-insensitive)."
            )
        df[target] = (df[target].astype(str).str.strip().str.lower() == "yes").astype(int)
```

Note the encode line gained `.str.strip()` to match the assertion's
normalization; for clean `yes`/`no` data the 0/1 output is identical.

**Regression test:** the validation suite (`tests/test_data_validation.py`)
already asserts bad target labels fail the gate; this assertion is the
belt-and-suspenders copy at the transformation boundary. The end-to-end test
confirms the clean path still trains.

---

## 5. `lru_cache` staleness on out-of-band retrain — comment only (by design)

**Where:** `src/loan_default/pipeline/prediction.py` — `_load_model()` (commit
`270a46a`).

**Root cause:** `_load_model()` is `@lru_cache(maxsize=1)`, so it holds the model
for the process lifetime. If `model.joblib` is rewritten out of band (e.g. a
separate `dvc repro`), a long-lived API process keeps serving the stale model.

**Decision — comment, not mtime invalidation.** The in-process `/train` endpoint
already calls `cache_clear()` after retraining, which is the supported refresh
path. Adding mtime-based invalidation would put a `stat()` on every prediction
and introduce a TOCTOU window for marginal benefit. We documented the contract
instead and extended `/train` to also clear the new `_feature_columns` cache:

```python
@lru_cache(maxsize=1)
def _load_model():
    # NOTE: this lru_cache holds the model for the process lifetime. An
    # out-of-band retrain that rewrites model.joblib will NOT be picked up until
    # the cache is cleared; the API's /train endpoint calls
    # ``_load_model.cache_clear()`` to do exactly that after retraining.
```

**Regression test:** none — this is a documentation + cache-clearing-wiring
change, exercised by the existing `/train` flow in the end-to-end test.

---

## Verified correct, not changed (the ML core)

These were audited and left untouched because they are already right:

- **Split before fit, fit on train only.** `data_transformation.transform()`
  calls `train_test_split(...)` first, then `preprocessor.fit(train_df[...])` on
  the training split only. No test rows reach the imputer/scaler/one-hot fit, so
  there is **no leakage**.
- **Stratified, reproducible split.** `train_test_split(stratify=df[target],
  random_state=self.config.random_state)` — class balance preserved and the split
  is deterministic.
- **Metrics on the held-out test set.** `model_evaluation.evaluate()` reads
  `test.csv` (never seen in training) and computes accuracy/precision/recall/f1
  on it; ROC-AUC is computed on **probabilities** (`predict_proba(...)[:, 1]`),
  not hard labels.
- **Serving/eval threshold parity.** Both the evaluator and `/predict` read
  `decision_threshold` from the same config, so the served decision boundary
  matches the one the reported metrics were computed at.
- **One serving artifact.** Training bundles preprocessor + model into a single
  `Pipeline`, so `/predict` applies the exact training transforms.
