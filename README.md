# Trade Index ML App

A machine-learning service that predicts New Zealand overseas trade index
values from period and series metadata, built end to end with DevOps and MLOps
practices: version-controlled data, tracked experiments, a containerised
inference API, and separate CI and delivery pipelines.

| | |
|---|---|
| **Task** | Regression — predict `Data_value` for one trade-index series and quarter |
| **Model** | `RandomForestRegressor`, R² **0.9085** / RMSE **240.03** on a 20 % hold-out |
| **Baseline** | `LinearRegression`, R² 0.4689 / RMSE 578.18 |
| **Dataset** | Stats NZ Overseas Trade Indexes, 104,613 raw rows |
| **API** | FastAPI, `GET /health` and `POST /predict` |
| **Image** | `python:3.12-slim`, 181 MB compressed |
| **Tests** | 96, including multi-case boundary coverage |

---

## Table of contents

- [How code moves to production](#how-code-moves-to-production)
- [Branching and pipelines](#branching-and-pipelines)
- [CI pipeline](#ci-pipeline)
- [CD pipeline](#cd-pipeline)
- [Data versioning (DVC)](#data-versioning-dvc)
- [Experiment tracking (MLflow)](#experiment-tracking-mlflow)
- [Testing](#testing)
- [Running locally](#running-locally)
- [Project structure](#project-structure)

---

## How code moves to production

```text
feature branch
     │  git commit / git push
     ▼
   dev  ──── pull request ───▶ CI: lint + format + 96 tests + docker build
     │                          (a red CI blocks the merge)
     │  merge
     ▼
 staging ── pull request ───▶ CI again on the PR
     │  merge
     │                        CD: staging .env from the staging Environment
     │                            → docker build trade-index:staging
     ▼
   main  ─── pull request ───▶ CI again on the PR
     │  merge
     │                        CD: production .env from the production Environment
     │                            → docker build trade-index:production
     ▼
  image built and tagged; deployment to a server is out of scope for this project
```

Every step is automated. A human only reviews and clicks merge.

---

## Branching and pipelines

Merging path: **`feature/*` → `dev` → `staging` → `main`**. A feature never goes
straight to `main`.

| Branch | Role | Trigger | Pipeline |
|---|---|---|---|
| `feature/*` | isolated work | — | none (local only) |
| `dev` | integration | pull request into it | **CI** |
| `dev` | integration | push to it | **CI** |
| `staging` | release candidate | pull request into it | **CI** |
| `staging` | release candidate | merge (push) to it | **CD — staging image** |
| `main` | production | pull request into it | **CI** |
| `main` | production | merge (push) to it | **CD — production image** |

Branch protection is enabled on `dev`, `staging` and `main`:

- a pull request is required (no direct pushes),
- the `Test, lint and build` status check must pass,
- the branch must be up to date before merging.

Together these enforce the brief's rule: *a PR cannot be merged if CI fails.*

---

## CI pipeline

`.github/workflows/ci.yml` — one job, `Test, lint and build`, on `ubuntu-latest`.
Runs on **pull requests targeting `dev`, `staging` or `main`**, and on pushes to
`dev`.

| Step | Command | Why |
|---|---|---|
| Install | `pip install -r requirements-dev.txt` | runtime + test + lint dependencies |
| Lint | `ruff check app ml tests` | static analysis |
| Format | `ruff format --check app ml tests` | consistent formatting |
| Test | `pytest -q` | 96 tests, all edge cases |
| Build | `docker build -t trade-index:ci .` | the image must build |

Nothing is pushed from CI. Publishing is the CD workflow's job.

Local equivalent:

```bash
ruff check app ml tests
ruff format --check app ml tests
pytest -q
docker build -t trade-index:ci .
```

---

## CD pipeline

`.github/workflows/cd.yml` — two jobs, each gated on a branch and bound to a
GitHub **Environment**:

| Job | Condition | Environment | Produces |
|---|---|---|---|
| `build-staging` | `github.ref == 'refs/heads/staging'` | `staging` | `trade-index:staging`, `trade-index:staging-<sha>` |
| `build-production` | `github.ref == 'refs/heads/main'` | `production` | `trade-index:production`, `trade-index:production-<sha>` |

Using `environment:` makes GitHub resolve `secrets.*` from that environment's
own secret store, so the two pipelines are configured independently even though
the workflow file is shared.

Each job:

1. writes a `.env` from the environment's secrets,
2. sources that `.env`,
3. passes its values to `docker build` as `--build-arg`,
4. tags the image twice — a moving tag and a commit-specific tag.

The `.env` values become build arguments rather than a copied file, so no secret
file is ever stored in an image layer. See [DEPLOYMENT.md](DEPLOYMENT.md) for the
setup and the trade-offs.

---

## Data versioning (DVC)

Raw data is **never committed to Git**. DVC stores the payload in Azure Blob
Storage and Git keeps only lightweight `.dvc` pointer files.

| Version | File | Rows | DVC md5 | Git tag |
|---|---|---|---|---|
| v1 — raw | `data/raw/raw_dataset.csv` | 104,613 | `42171ec985af1d3aed38f4d6f497030b` | `data-v1` |
| v2 — cleaned | `data/processed/clean_v2.csv` | 77,337 | `a0f682da4b534942450d8f30ba6cca1d` | `data-v2` |

Remote: `myazure` → `azure://dvc` on storage account `wut2026`, authenticated
with a SAS token stored only in `.dvc/config.local` (git-ignored).

```bash
dvc pull                     # materialise the tracked data
dvc status --cloud           # compare local cache with Azure
git switch --detach data-v1  # select a dataset version
dvc pull                     # get that version's bytes
```

Full details, including every change from v1 to v2 and the commit mapping, are
in [ml/README_data.md](ml/README_data.md).

---

## Experiment tracking (MLflow)

Two experiments on the v2 dataset, 80/20 hold-out, `random_state=42`:

| Run | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| `baseline-linear` | LinearRegression | 578.1820 | 324.5331 | 0.4689 |
| `tuned-forest` | RandomForestRegressor | **240.0274** | **154.4691** | **0.9085** |

Every run records the five items the brief requires: git commit, DVC data
hash, hyperparameters, metrics, and the fitted model artifact.

```bash
python -m ml.train                          # trains and logs both runs
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://127.0.0.1:5000
```

Rationale for choosing the forest and for optimising RMSE is in
[ml/README_experiments.md](ml/README_experiments.md).

> MLflow 3.x refuses the legacy `./mlruns` file store, so training writes to
> `sqlite:///mlflow.db`. It also needs `skops_trusted_types` to serialise tree
> ensembles; the training script passes it explicitly.

---

## Testing

96 tests across four files.

| File | Tests | Covers |
|---|---|---|
| `tests/unit/test_schemas.py` | 32 | year/month bounds, blank and missing required fields, extra fields, wrong types, optional-field normalisation, column order |
| `tests/unit/test_data_pipeline.py` | 31 | column renaming, leaky-column removal, unit filtering, constant-column removal, target coercion, null handling, period parsing, idempotency |
| `tests/integration/test_model_loader.py` | 8 | missing artifact, prediction before load, multi-row rejection, unseen categories, summary handling, reload |
| `tests/api/test_predict_api.py` | 25 | health, deterministic predictions, 6 kinds of invalid payload, 9 per-field omissions, empty body, unseen categories, 503 without a model, OpenAPI schema |

Notable boundary cases: years 1899/2100/2101/9999, months 0/−1/13/99, a string
where an int is required, an empty JSON body, a category never seen during
training, a malformed `Period` (`""`, `"2020"`, `"2020-06"`), and an empty frame
passed to the cleaning pipeline.

```bash
pytest -q
```

---

## Running locally

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
pip install -r requirements-dev.txt

python -m ml.data_pipeline          # build data/processed/clean_v2.csv
python -m ml.train                  # train, log to MLflow, export models/model.joblib

uvicorn app.main:app --reload       # http://127.0.0.1:8000/docs
```

### With Docker

```bash
docker build -t trade-index:local .
docker run --rm -p 127.0.0.1:8001:8000 -e APP_ENV=local trade-index:local
```

```bash
curl http://localhost:8001/health
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
        "year": 2024, "month": 6, "STATUS": "FINAL",
        "Subject": "Overseas Trade Indexes - Prices - OTP",
        "Group": "Export price indexes",
        "Series_title_1": "Price",
        "Series_title_2": "BEEF AND VEAL",
        "Series_title_3": "June 2002 Quarter Base (=1000)",
        "Series_title_4": "Trend"
      }'
```

Expected prediction: `1366.61`.

---

## Project structure

```text
.
├── app/
│   ├── main.py              # FastAPI app, /health and /predict, lifespan model load
│   ├── model_loader.py      # loads the pipeline once, exposes predict()
│   └── schemas.py           # request/response models and all input constraints
├── ml/
│   ├── data_pipeline.py     # v1 → v2 cleaning and feature engineering
│   ├── train.py             # trains both models, logs to MLflow, exports the artifact
│   ├── configs/baseline.yaml
│   ├── README_data.md       # dataset versions and commit mapping
│   └── README_experiments.md
├── tests/
│   ├── unit/                # schema validation
│   ├── integration/         # model loading and prediction
│   └── api/                 # HTTP surface
├── models/
│   ├── model.joblib         # serving artifact (compressed, ~15 MB)
│   └── run_summary.json     # commit / data version / model type
├── data/                    # DVC-managed, not committed
├── scripts/
│   └── verify_project.py    # one-command acceptance check
├── Dockerfile
├── .dockerignore
├── pyproject.toml           # ruff + pytest configuration
├── requirements.txt         # runtime only
├── requirements-dev.txt     # + pytest, httpx, ruff
├── requirements-train.txt   # + mlflow, pyyaml
├── DEPLOYMENT.md
├── COMPLIANCE.md            # brief requirement → implementation map
└── .github/workflows/{ci.yml,cd.yml}
```

`requirements.txt` intentionally excludes MLflow and PyYAML: they are needed for
training but not for serving, and leaving them out keeps the image at 181 MB
instead of roughly 400 MB.

Run `python scripts/verify_project.py` to check the whole deliverable at once;
[COMPLIANCE.md](COMPLIANCE.md) explains what each requirement maps to.
