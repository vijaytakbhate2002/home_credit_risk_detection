# Home Credit Default Risk — Modeling & Serving

End-to-end project for the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) Kaggle problem. It covers the full lifecycle: pulling the raw competition tables, aggregating and selecting features, training a LightGBM default-probability model, packaging the model + preprocessing into a single artifact, and serving predictions through a FastAPI service.

The project is organized in three parts:

1. **[Part 1 — LightGBM Model Training](#part-1--lightgbm-model-training)** — data extraction, feature selection, and model training.
2. **[Part 2 — FastAPI Integration](#part-2--fastapi-integration)** — how the trained model is served as a REST API.
3. **[Part 3 — Deployment (AWS) — WIP](#part-3--deployment-aws--work-in-progress)** — planned cloud deployment.

**Target:** predict whether a client will have payment difficulties (`TARGET` in `application_train.csv`, bad rate ≈ 8%).

## Project structure

```
home_credit_risk_modeling/
├── app.py                          # FastAPI app (single / batch-json / batch-csv endpoints)
├── main.py                         # CLI batch prediction pipeline over the test set
├── requirements.txt
├── modeling/
│   ├── 0_data_extraction.ipynb     # Kaggle download → local CSVs
│   ├── 1_feature_selection.ipynb   # Aggregation, merge, feature selection
│   ├── 2_data_uploader.ipynb       # Selected features → PostgreSQL (Neon)
│   ├── 3_lightgbm_training.ipynb   # Preprocessing + LightGBM training + artifact dump
│   ├── predictor.py                # ModelPredictor: loads artifact, runs inference
│   ├── preprocessor.py             # DataPreprocessor: impute / encode / scale / align
│   ├── model_artifacts/            # Saved .pkl artifacts (model + metadata + preprocessing)
│   ├── input_data/                 # Raw / aggregated / feature-selected data (DVC tracked)
│   └── submissions/                # Kaggle submission CSVs
└── test_api/
    ├── api_prediction_test.ipynb   # Hits the running API and compares vs submission
    └── sample_single_request.json  # Example single-row request payload
```

Large CSV inputs are tracked with **DVC** (`*.dvc` pointer files) and kept out of git.

---

# Part 1 — LightGBM Model Training

This part is implemented across three notebooks under `modeling/`: `0_data_extraction.ipynb`, `1_feature_selection.ipynb`, and `3_lightgbm_training.ipynb`. The steps below describe exactly what each notebook does.

## Step 0 — Data Extraction (`0_data_extraction.ipynb`)

1. Download the full competition bundle with `kagglehub.competition_download('home-credit-default-risk')` (~688 MB).
2. Copy every downloaded file into `input_data/row_input_data/`.
3. Print a directory listing with per-file sizes to confirm all tables were copied.

Resulting raw tables and shapes:

| Table | Shape |
|-------|-------|
| `application_train.csv` | (307511, 122) |
| `application_test.csv` | (48744, 121) |
| `bureau.csv` | (1716428, 17) |
| `bureau_balance.csv` | (27299925, 3) |
| `credit_card_balance.csv` | (3840312, 23) |
| `installments_payments.csv` | (13605401, 8) |
| `POS_CASH_balance.csv` | (10001358, 8) |
| `previous_application.csv` | (1670214, 37) |
| `HomeCredit_columns_description.csv` | (219, 5) |
| `sample_submission.csv` | (48744, 2) |

## Step 1 — Feature Selection & Engineering (`1_feature_selection.ipynb`)

The grain of the model is one row per loan application, keyed by `SK_ID_CURR`. Auxiliary tables (bureau, credit card, installments, etc.) are at a finer grain, so they are aggregated up to `SK_ID_CURR` and merged onto the application tables, then funneled through three statistical filters.

### 1.1 Load & key validation
- Read all raw tables from `input_data/row_input_data/`.
- Validate the foreign keys (`SK_ID_CURR`, `SK_ID_BUREAU`, `SK_ID_PREV`) by checking total vs unique counts for each table.

### 1.2 First-level aggregation
- Build per-table aggregation dictionaries automatically: **numeric** columns get `mean`, `median`, `max`, `min`; **categorical** (object) columns get `mode` (first mode of non-null values).
- Aggregate each auxiliary table with `aggregate_dataset(df, agg_dict, groupby_col)` using these group keys:
  - `bureau` → `SK_ID_CURR`
  - `bureau_bal` → `SK_ID_BUREAU`
  - `cc_bal`, `inst_pmt`, `pos_cash` → `SK_ID_PREV`
  - `prev_app` → `SK_ID_CURR`
- Resulting aggregate column names follow `{COLUMN}_{FUNC}` (uppercased), e.g. `DAYS_CREDIT_MEAN`.
- Save the six aggregate tables to `input_data/row_agg_data/`: `bureau_agg.csv`, `bureau_bal_agg.csv`, `cc_bal_agg.csv`, `inst_pmt_agg.csv`, `pos_cash_agg.csv`, `prev_app_agg.csv`. (Full-dataset aggregation takes ~45–50 min.)

### 1.3 Second-level aggregation + merge onto applications
- Tables keyed by `SK_ID_PREV` / `SK_ID_BUREAU` are rolled up a second time to `SK_ID_CURR` (`_second_level_agg`: numeric → mean, non-numeric → mode). For `bureau_bal_agg`, `SK_ID_BUREAU` is first mapped back to `SK_ID_CURR` via the `bureau` table.
- Each aggregate block is suffixed by its source (e.g. `_bureau`, `_prev_app`) and left-joined onto `application_train` / `application_test` on `SK_ID_CURR`.
- Save merged tables to `input_data/row_merged_data/`. Result: train `(307511, 404)`, test `(48744, 403)`.

### 1.4 Three-stage feature selection funnel
Applied on the merged training table (starting from **403** features):

1. **Fill-rate filter** — drop columns with fill rate `< 5%`. → 8 dropped (395 left).
2. **Near-zero-variance filter** — drop columns where a single value covers `≥ 99%` of rows (or only one unique value). → 18 dropped (377 left).
3. **Information Value (IV) filter** — on a stratified 100k-row sample, fit `optbinning.OptimalBinning` per feature, compute IV from the binning table, and drop features with `IV < 0.02`. → 69 dropped (308 left).

Drop funnel summary:

| Stage | Feature count | Dropped |
|-------|---------------|---------|
| Initial (merged) | 403 | — |
| After fill rate | 395 | 8 |
| After NZV | 377 | 18 |
| After IV | 308 | 69 |

- Save the final feature-selected tables to `input_data/row_fs_data/`: `app_train_fs.csv` (309 cols incl. `SK_ID_CURR` + `TARGET`) and `app_test_fs.csv` (308 cols).

> `2_data_uploader.ipynb` is an optional side step that uploads `app_train_fs.csv` to a PostgreSQL (Neon) database via SQLAlchemy `to_sql` — it is not required for training.

## Step 2 — LightGBM Training (`3_lightgbm_training.ipynb`)

### 2.1 Load feature-selected data
- Read `app_train_fs.csv` `(307511, 309)` and `app_test_fs.csv` `(48744, 308)` from `input_data/row_fs_data/`.
- Confirm bad rate ≈ 8%.

### 2.2 Preprocessing (fit on train, apply to test)
Done in this order, with the fitted transformers retained for serving:

1. **Imputation** — `SimpleImputer(strategy='median')` for numeric (282 cols), `SimpleImputer(strategy='most_frequent')` for categorical (26 cols).
2. **Encoding** — `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` on categorical columns.
3. **Scaling** — `MinMaxScaler` on all feature columns (everything except `SK_ID_CURR` and `TARGET`).

(An optional WoE/IV transformation via `optbinning` is included behind a `WOE_TRANSFORMATION` flag but is disabled for the final model.)

### 2.3 Train/validation model
- Split: `train_test_split(test_size=0.2, stratify=y, random_state=42)`.
- Handle class imbalance with `scale_pos_weight = n_negative / n_positive` (≈ 11.39).
- Train `lgb.LGBMClassifier` with early stopping. Key params:

  ```python
  LIGHTGBM_PARAMS = {
      'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
      'n_estimators': 10000, 'learning_rate': 0.05, 'num_leaves': 31,
      'max_depth': -1, 'min_child_samples': 20,
      'subsample': 0.8, 'colsample_bytree': 0.8,
      'scale_pos_weight': scale_pos_weight, 'random_state': 42,
      'n_jobs': -1, 'verbosity': -1,
  }
  # fit with: early_stopping(stopping_rounds=100), eval_metric='auc'
  ```

- Validation results (best iteration **370**):

  | Metric | Train | Valid |
  |--------|-------|-------|
  | AUC | 0.857571 | 0.782624 |
  | Gini | 0.715142 | 0.565248 |

  AUC gap ≈ 0.075 (benchmark target was 0.80).

### 2.4 Final model + artifact dump
- Retrain on the **full** training set with `n_estimators = best_iteration` (370).
- Bundle everything needed for serving into one dict and persist with `joblib.dump(..., compress=3)` to `model_artifacts/home_credit_lgbm_<timestamp>.pkl`:

  ```python
  model_artifact = {
      "model": final_model,
      "metadata": {model_name, created_at, target_column, input_features,
                   feature_dtypes, output_schema, model_metrics, library_versions},
      "preprocessing": {num_imputer, cat_imputer, encoder, scaler},
  }
  ```

  Storing the fitted preprocessing transformers inside the artifact is what lets the API reproduce the exact training-time transformations at inference.

### 2.5 Analysis (optional cells)
- Build a Kaggle submission (`SUBMISSION` flag) from `predict_proba` on the test set.
- Inspect top features by LightGBM **gain** and by **SHAP** values; plot the validation **ROC curve** (best threshold ≈ 0.478).

---

# Part 2 — FastAPI Integration

The trained artifact is served through a FastAPI app (`app.py`) that reuses the same preprocessing/inference code written for training. The serving layer is built from two reusable classes plus the app itself.

## Serving components

### `modeling/preprocessor.py` — `DataPreprocessor`
Encapsulates imputation, encoding, scaling, and column alignment so inference matches training exactly.
- `fit(df)` — identifies numeric vs categorical columns (excluding `SK_ID_CURR` / `TARGET`) and fits the median imputer, most-frequent imputer, ordinal encoder, and MinMax scaler.
- `transform(df)` — applies impute → encode → scale in sequence. Crucially, `_align_and_prepare_columns` **pads any missing columns with NaN and reorders** to the exact set each transformer expects, so partial/arbitrary inputs still work.
- `create_preprocessor_from_artifact(path)` — rebuilds a fully-fitted `DataPreprocessor` directly from the saved model artifact's `preprocessing` block (reading the imputers' `feature_names_in_` to recover the exact column lists).

### `modeling/predictor.py` — `ModelPredictor`
Wraps the model + preprocessor for end-to-end inference.
- `__init__(model_path)` — loads the artifact with `joblib`, extracts `model`, `metadata`, and (if not supplied) builds the preprocessor from the artifact.
- `preprocess(df)` — drops `TARGET` if present, transforms via the preprocessor, then drops the ID column.
- `predict(df, return_proba=True, include_id=True, id_col='SK_ID_CURR')` — preprocesses, selects/reorders the expected feature set, validates none are missing, and returns a dataframe with `SK_ID_CURR` + `prediction_probability` (uses `best_iteration_`).
- Helpers: `predict_batch`, `get_model_info`, `get_feature_names`, `validate_input`.

## API application (`app.py`)
- Created as `FastAPI(title="Home Credit Risk Modeling API", version="1.0.0")`.
- **Startup hook** (`@app.on_event("startup")`) scans `modeling/model_artifacts/` for `.pkl` files, picks the **latest by modification time**, and loads it once into a global `ModelPredictor`, so the model is loaded a single time rather than per request.

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Health check + model metadata (name, created_at, num features, target, metrics). Returns 503 if the model isn't loaded. |
| `POST` | `/predict/single` | Single-row, real-time prediction from a JSON object of features. |
| `POST` | `/predict/batch/json` | Batch prediction from a JSON **list** of feature objects. |
| `POST` | `/predict/batch/csv` | Batch prediction from an uploaded **CSV file**; streams a CSV back as a download. |

Behavior shared across prediction endpoints:
- If `SK_ID_CURR` is missing, a placeholder ID is generated (single: `999999`; batch: a `range(100000, ...)`), so callers don't have to supply it.
- Missing features are auto-padded with NaN by the preprocessor (see `sample_single_request.json` — many fields are `null`).
- A `prediction_class` is derived from `prediction_probability >= 0.5`.
- Errors return `HTTP 400` with a descriptive message; an unloaded model returns `HTTP 503`.

## Running the API

```bash
pip install -r requirements.txt
# from the project root (so modeling/model_artifacts is discoverable):
python app.py
# or:
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Interactive docs are then available at `http://127.0.0.1:8000/docs`.

### Example: single prediction

```bash
curl -X POST http://127.0.0.1:8000/predict/single \
  -H "Content-Type: application/json" \
  -d @test_api/sample_single_request.json
```

Response shape:

```json
{ "SK_ID_CURR": 100005, "prediction_probability": 0.55, "prediction_class": 1 }
```

### Example: CSV batch

```bash
curl -X POST http://127.0.0.1:8000/predict/batch/csv \
  -F "file=@modeling/input_data/row_fs_data/app_test_fs.csv" \
  -o predictions.csv
```

## Testing the API & CLI
- **`test_api/api_prediction_test.ipynb`** loads `app_test_fs.csv`, sends the first rows to `/predict/single` (converting `NaN` → `None` for JSON), collects predictions, and compares them against `modeling/submissions/submission_v2.csv` (MSE / MAE / RMSE / correlation) to confirm the served predictions match the offline model.
- **`main.py`** is a standalone CLI pipeline that loads the latest artifact, validates the test set, runs `ModelPredictor.predict` over `app_test_fs.csv`, prints summary stats, and writes `test_predictions/predictions_<timestamp>.csv`.

---

# Part 3 — Deployment (AWS) — Work In Progress

> **WIP — not yet implemented.** The plan is to deploy the FastAPI service on **AWS**.

Planned direction (subject to change):

- Containerize the FastAPI app (Docker) and run it via a container service (e.g. ECS/Fargate or App Runner) behind an HTTPS endpoint.
- Store / retrieve the model artifact from **S3** (and/or pull versioned data via DVC remote on S3).
- Add CI/CD for build + deploy, health checks against `GET /`, and basic autoscaling.

Details (infrastructure-as-code, IAM, networking, monitoring) will be filled in once implementation begins.

---

## Prerequisites & setup (training)

- Python 3.10+
- [Kaggle API credentials](https://www.kaggle.com/docs/api) for `kagglehub` (Step 0)
- (Optional) PostgreSQL / Neon connection string in `.env` as `DATABASE_URL` for `2_data_uploader.ipynb`
- DVC (artifacts/data are DVC-tracked)

```bash
git clone <your-repo-url>
cd home_credit_risk_modeling
python -m venv env
source env/bin/activate        # Windows: .\env\Scripts\activate
pip install -r requirements.txt
```

Then run the notebooks under `modeling/` in order (`0` → `1` → `3`; `2` is optional) to reproduce the artifact, or use the committed artifact in `modeling/model_artifacts/` to serve directly.

## Key libraries

`lightgbm==4.6.0`, `scikit-learn==1.5.2`, `optbinning==0.20.0`, `shap==0.52.0`, `pandas`, `numpy`, `joblib==1.5.3`, `kagglehub==0.3.5`, `dvc==3.67.1`, `fastapi`, `uvicorn`, `python-multipart` (full pins in `requirements.txt`).
