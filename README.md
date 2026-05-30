# Home Credit Default Risk — Modeling Pipeline

End-to-end exploratory and feature-engineering workflow for the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) Kaggle competition. Raw competition tables are downloaded, loaded into PostgreSQL, aggregated at the loan level (`SK_ID_CURR`), and prepared for modeling.

## Overview

| Notebook | Purpose |
|----------|---------|
| `data_extraction.ipynb` | Download competition data via `kagglehub` and copy CSVs into the project |
| `data_uploader.ipynb` | Load raw CSVs into PostgreSQL (`pandas.to_sql`) |
| `data_processing.ipynb` | Connect with Jupyter SQL magic and query tables in the database |
| `feature_selection.ipynb` | Aggregate auxiliary tables, save engineered features, merge & select features (in progress) |

**Target:** predict whether a client will have payment difficulties (`TARGET` in `application_train.csv`).

## Project structure

```
home_credit_risk_modeling/
├── data_extraction.ipynb      # Kaggle download → local CSVs
├── data_uploader.ipynb        # CSVs → PostgreSQL
├── data_processing.ipynb      # SQL exploration in Jupyter
├── feature_selection.ipynb    # Aggregations, merges, feature selection
├── requirements.txt
├── .env                       # DATABASE_URL (not committed)
├── input_data/
│   ├── row_input_data/        # Raw competition CSVs (~688 MB download)
│   └── row_agg_data/          # Aggregated feature tables (generated)
└── env/                       # Virtual environment (gitignored)
```

Raw and aggregated CSV files are **gitignored** (see `.gitignore`).

## Prerequisites

- Python 3.10+ (project uses a local `env` virtual environment)
- [Kaggle API credentials](https://www.kaggle.com/docs/api) (`~/.kaggle/kaggle.json` or env vars) for `kagglehub`
- PostgreSQL instance and a connection string

### Extra packages (used in notebooks, not all in `requirements.txt`)

```bash
pip install python-dotenv ipython-sql
```

## Setup

1. **Clone and create a virtual environment**

   ```bash
   git clone <your-repo-url>
   cd home_credit_risk_modeling
   python -m venv env
   # Windows
   .\env\Scripts\activate
   # macOS/Linux
   source env/bin/activate
   pip install -r requirements.txt
   pip install python-dotenv ipython-sql
   ```

2. **Configure the database**

   Create a `.env` file in the project root:

   ```env
   DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DATABASE
   ```

3. **Configure Kaggle** (for `data_extraction.ipynb`)

   Place `kaggle.json` under `~/.kaggle/` or set `KAGGLE_USERNAME` / `KAGGLE_KEY`.

## Workflow (run notebooks in order)

### 1. Data extraction

Run `data_extraction.ipynb` to:

- Download all competition files with `kagglehub.competition_download('home-credit-default-risk')`
- Copy them to `input_data/row_input_data/`

Expected files include: `application_train.csv`, `application_test.csv`, `bureau.csv`, `bureau_balance.csv`, `credit_card_balance.csv`, `installments_payments.csv`, `POS_CASH_balance.csv`, `previous_application.csv`, `HomeCredit_columns_description.csv`, `sample_submission.csv`.

### 2. Data upload

Run `data_uploader.ipynb` to load every CSV in `input_data/row_input_data/` into PostgreSQL (table name = filename without extension, `if_exists='replace'`, `chunksize=5000`).

> Large tables (e.g. `bureau_balance`, `installments_payments`) can take a long time and may need tuning (chunk size, indexes, or loading only subsets).

### 3. Data processing

Run `data_processing.ipynb` to connect with `%load_ext sql` and run SQL against the loaded tables (e.g. `SELECT * FROM public.application_test LIMIT 10`).

### 4. Feature selection & engineering

Run `feature_selection.ipynb` to:

1. Load raw tables from `input_data/row_input_data/`
2. Validate foreign keys (`SK_ID_CURR`, `SK_ID_BUREAU`, `SK_ID_PREV`)
3. **Aggregate** bureau, bureau balance, credit card balance, installments, POS/cash, and previous applications by `SK_ID_CURR` (mean, median, max, min for numeric columns; mode for categoricals)
4. Save aggregates to `input_data/row_agg_data/`:
   - `bureau_agg.csv`, `bureau_bal_agg.csv`, `cc_bal_agg.csv`, `inst_pmt_agg.csv`, `pos_cash_agg.csv`, `prev_app_agg.csv`
5. **Merge** aggregated features onto `application_train` / `application_test` and perform feature selection *(section in progress)*

Aggregation on the full dataset can take **several hours** on a typical machine.

## Data dictionary

Column definitions are in `HomeCredit_columns_description.csv` (included in the competition download). Main tables:

| Table | Grain | Join key to applications |
|-------|--------|---------------------------|
| `application_train` / `application_test` | One row per loan | `SK_ID_CURR` |
| `bureau` | Credit bureau history | `SK_ID_CURR` |
| `bureau_balance` | Monthly bureau status | `SK_ID_BUREAU` → bureau |
| `previous_application` | Prior Home Credit apps | `SK_ID_CURR` |
| `credit_card_balance`, `installments_payments`, `POS_CASH_balance` | Monthly behavior | `SK_ID_CURR` (via `SK_ID_PREV`) |

## Tech stack

- **Python:** pandas, NumPy, Jupyter
- **Data:** Kaggle (`kagglehub`), local CSV storage
- **Database:** PostgreSQL (`psycopg2`, SQLAlchemy, `ipython-sql`)

## Status & next steps

- [x] Download and stage raw data
- [x] PostgreSQL upload pipeline
- [x] Per-table aggregations and CSV export
- [ ] Complete table merging and feature selection
- [ ] Train/evaluate models and generate submission

## License & data

Competition data is subject to [Kaggle competition rules](https://www.kaggle.com/competitions/home-credit-default-risk/rules). Use only for learning and competition participation.

## Acknowledgments

- [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) on Kaggle
