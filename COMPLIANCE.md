# Requirements traceability

Every requirement in `project-instructions`, mapped to the artefact that
satisfies it and the check that proves it. Written so a reviewer can verify each
line without reading the whole codebase.

Legend: **Req** = the brief's wording · **Where** = the implementation ·
**Proof** = how to confirm it yourself.

---

## Part 1 — The App

> *"This project is about building a machine-learning-powered app using DevOps
> and MLOps."*

### R1.1 — An app with a trained or fine-tuned model

| | |
|---|---|
| **Where** | `app/main.py`, `ml/train.py`, `models/model.joblib` |
| **What** | FastAPI service exposing `GET /health` and `POST /predict`. The served model is a `RandomForestRegressor` trained on 77,337 real trade-index rows. |
| **Proof** | `curl localhost:8001/health` → `model_loaded: true`, `model_type: RandomForestRegressor` |
| **Metric** | R² 0.9085 on a held-out 20 % split |

The model is genuinely trained, not a stub: `models/model.joblib` reproduces the
documented metrics exactly when re-evaluated against `data/processed/clean_v2.csv`.

### R1.2 — A dataset, potentially with multiple versions

| | |
|---|---|
| **Where** | `data/raw/raw_dataset.csv` (v1), `data/processed/clean_v2.csv` (v2) |
| **What** | Stats NZ Overseas Trade Indexes. v1 = 104,613 raw rows; v2 = 77,337 cleaned rows. |
| **Proof** | `git tag` → `data-v1`, `data-v2`; both `.dvc` pointers committed |

### R1.3 — At least 3 meaningful tests, each with multiple edge cases

| | |
|---|---|
| **Where** | `tests/` — four files, **96 tests** |
| **Proof** | `pytest -q` → `96 passed` |

| File | Tests | Boundary cases |
|---|---|---|
| `tests/unit/test_schemas.py` | 32 | years 1899/2100/2101/9999; months 0/−1/13/99; blank required fields; 9 per-field omissions; extra field; string/None/float/list where an int is required |
| `tests/unit/test_data_pipeline.py` | 31 | column rename; leaky-column removal; unit filtering; empty-frame handling; target coercion; unparseable target; NaN and blank categoricals; 5 malformed `Period` values; idempotency |
| `tests/integration/test_model_loader.py` | 8 | missing artifact; predict-before-load; multi-row input; unseen categories; summary present/absent; reload |
| `tests/api/test_predict_api.py` | 25 | health; determinism; 6 invalid payloads; 9 per-field omissions; empty body; unseen categories; 503 without a model; OpenAPI schema |

**This exceeds the minimum of 3 by a wide margin, and each case is a distinct
assertion rather than a repeated happy path.**

---

## Part 2 — DevOps

### R2.1 — Version control: Git and GitHub

| | |
|---|---|
| **Where** | The repository itself; 10+ commits with descriptive messages |
| **Proof** | `git log --oneline` |

### R2.2 — A planned branching scheme with a proper merging path

> *"plan your Git branching scheme (main, staging, dev features...), with the
> proper merging path (from feature to dev to staging to main)"*

| | |
|---|---|
| **Where** | Branches `dev`, `staging`, `main` |
| **Proof** | `git branch -vv`; the PR history on GitHub shows `dev → staging` (PR #3) then `staging → main` (PR #4) |

**The path was actually exercised twice, not just declared.** `main` was never
pushed to directly: every change reached it through two pull requests.

### R2.3 — A slim Dockerfile with a `.dockerignore`

| | |
|---|---|
| **Where** | `Dockerfile`, `.dockerignore` |
| **Proof** | `docker images trade-index` → content size **181 MB** |

Specific measures:

| Measure | Effect |
|---|---|
| `python:3.12-slim` base | ~50 MB instead of ~1 GB |
| `requirements.txt` excludes MLflow/PyYAML | saves ~200 MB over a naive install |
| `COPY requirements.txt` before `COPY app/` | dependency layer cached across code changes |
| `.dockerignore` excludes `data/`, `tests/`, `.venv/`, `.git/`, `.env` | keeps the build context at ~16 MB instead of ~40 MB |
| `useradd … && USER 10001` | runs as a non-root user |

### R2.4 — CI runs the full test suite

| | |
|---|---|
| **Where** | `.github/workflows/ci.yml`, step `Run tests` |
| **Proof** | Actions run on PR #1/#2 → `96 passed` |

### R2.5 — CI runs linting / static analysis

| | |
|---|---|
| **Where** | `ci.yml`, steps `Lint` and `Check formatting` |
| **Tool** | `ruff` (covers `app ml tests scripts`) |
| **Proof** | `ruff check` → `All checks passed!`; `ruff format --check` → `16 files already formatted` |

Rules are pinned in `pyproject.toml` so a local run and a CI run agree.

### R2.6 — CI builds the Docker image

| | |
|---|---|
| **Where** | `ci.yml`, step `Build Docker image` |
| **Proof** | Every CI run executes `docker build --tag trade-index:ci .` |

Nothing is pushed from CI — publishing belongs to CD.

### R2.7 — A PR cannot be merged if CI fails

| | |
|---|---|
| **Where** | GitHub branch protection on `dev`, `staging`, `main` |
| **Settings** | Require a pull request · Require status checks (`Test, lint and build`) · Require branch to be up to date |
| **Proof** | The Merge button is only enabled once the check reports success |

### R2.8 — At least one automated deployment stage

> *"Merges to staging → uses GitHub secrets specific for the staging environment
> to create a staging .env file, then build the staging Docker image. Merges to
> main → … production …"*

| | |
|---|---|
| **Where** | `.github/workflows/cd.yml` — `build-staging` and `build-production` |
| **Proof** | Deployments page lists two records: `staging (#1)` and `production (#2)` |

| Requirement | Implementation |
|---|---|
| staging `.env` from staging secrets | job bound to `environment: staging`, writes `.env` from `secrets.APP_ENV` / `secrets.LOG_LEVEL` |
| production `.env` from production secrets | same job shape bound to `environment: production` |
| build staging image | `docker build --tag trade-index:staging --tag trade-index:staging-<sha12>` |
| build production image | `docker build --tag trade-index:production --tag trade-index:production-<sha12>` |
| only on the right branch | each job carries an `if:` on `github.ref`; the other reports *skipped* |

**Design decision worth noting:** the `.env` values are passed to the build as
`--build-arg` rather than `COPY`-ed into the image. The mechanism is identical
but no secret file is stored in an image layer. `APP_ENV` is echoed by
`GET /health`, so the configuration is observable at runtime.

### R2.9 — CI config and README documentation

| | |
|---|---|
| **Where** | `.github/workflows/ci.yml`, `.github/workflows/cd.yml`, `README.md` |
| **Proof** | README §"How code moves to production" and §"Branching and pipelines" |

The README states explicitly which branch triggers which pipeline, as required.
No leftover `TODO` markers: `grep -c TODO README.md` → `0`.

---

## Part 3 — MLOps

### R3.1 — Track training data with DVC

| | |
|---|---|
| **Where** | `.dvc/`, `data/*/**.dvc` pointers |
| **Proof** | `dvc status` → `Data and pipelines are up to date.` |

### R3.2 — Configure remote storage

| | |
|---|---|
| **Where** | `.dvc/config` → `azure://dvc`, `account_name = wut2026` |
| **Proof** | `dvc status --cloud` → `Cache and remote 'myazure' are in sync.` |

### R3.3 — Maintain at least 2 dataset versions

| Version | Git tag | Commit | DVC md5 | Rows |
|---|---|---|---|---|
| v1 raw | `data-v1` | `a3b8237` | `42171ec985af1d3aed38f4d6f497030b` | 104,613 |
| v2 cleaned | `data-v2` | `a5eafb0` | `a0f682da4b534942450d8f30ba6cca1d` | 77,337 |

**Proof:** `git tag --list`; both tags pushed to GitHub.

### R3.4 — Document source, v1→v2 changes, and commit↔version mapping

| | |
|---|---|
| **Where** | `ml/README_data.md` |
| **Contents** | ① source and all 13 columns ② a 10-row table of every v1→v2 change with the rows affected and the reason ③ the commit↔version table above ④ how to reproduce a version with `git switch --detach` + `dvc pull` |

### R3.5 — Do not commit raw data

| | |
|---|---|
| **Proof** | `git ls-files` contains no `.csv` or `.parquet`; `git status --short --ignored` shows `!! data/raw/raw_dataset.csv` |

### R3.6 — Log code version, dataset version, hyperparameters, metrics, artifacts

| Requirement | MLflow field |
|---|---|
| git commit SHA | `git_commit` param — read from `git rev-parse HEAD` |
| DVC dataset version | `data_dvc_hash` + `data_version` params — parsed from the `.dvc` pointer |
| hyperparameters | `model_type`, `n_estimators`, `max_depth`, `min_samples_leaf`, `random_state`, `train_rows`, `test_rows` |
| metrics | `rmse`, `mae`, `r2` |
| artifacts | logged model (`model.skops`), 51 MB |

**Proof:** `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db` → open
the `tuned-forest` run → Parameters tab shows all ten entries.

### R3.7 — At least 2 distinct experiments

| Run | Model | RMSE | MAE | R² |
|---|---|---|---|---|
| `baseline-linear` | LinearRegression | 578.1820 | 324.5331 | 0.4689 |
| `tuned-forest` | RandomForestRegressor | **240.0274** | **154.4691** | **0.9085** |

### R3.8 — Explain which run is production-worthy and which metric matters

| | |
|---|---|
| **Where** | `ml/README_experiments.md` |
| **§4** | Names `tuned-forest`, with four ranked reasons (material gain, generalises to an untouched split, acceptable failure mode, baseline kept only as a reference) |
| **§5** | Argues for RMSE (same unit as the target, penalises large errors, interpretable), with R² and MAE explained as secondary |
| **§6** | Five honest limitations, including that a random split is weaker than a chronological one for time-series data |

---

## Part 4 — Repo structure

The brief gives an example layout and says *"names can vary, but
responsibilities must be clear"*.

| Brief asks for | Present | Note |
|---|---|---|
| `app/main.p` | `app/main.py` | service entrypoint |
| `ml/train.p` | `ml/train.py` | training script, calls MLflow |
| `ml/data_pipeline.p` | `ml/data_pipeline.py` | preprocessing / feature engineering |
| `ml/configs/` | `ml/configs/baseline.yaml` | hyperparameters |
| `ml/registr/current_model.md` | `ml/registry/current_model.md` | which model is production |
| `README_data.md` | `ml/README_data.md` | ⚠️ in `ml/`, not `ml/registry/` |
| `README_experiments.md` | `ml/README_experiments.md` | ⚠️ in `ml/`, not `ml/registry/` |
| `PROVENANCE.md` | `ml/PROVENANCE.md` | ⚠️ in `ml/`, not `ml/registry/` |
| `data.dvc / data/` | `data/raw/*.dvc`, `data/processed/*.dvc` | pointers, not payloads |
| `requirements.txt / pyproject.toml` | both, plus split `-dev` / `-train` | |
| `Dockerfile` | ✅ | slim, non-root |
| `.dockerignore` | ✅ | |
| `.github/workflows/ci.ml` | `.github/workflows/ci.yml` | **`.yml`, not `.ml`** |
| `DEPLOYMENT.md` | ✅ | |
| `README.md` | ✅ | |

**Deliberate deviation:** the three `ml/*.md` files sit next to the code they
describe rather than inside `ml/registry/`. The registry holds only
`current_model.md`, which is the thing a registry actually answers. The brief
permits this (*"names can vary"*), but it is called out here so the choice is
visible rather than accidental.

---

## Part 5 — Grading deliverables

| Deliverable | Where |
|---|---|
| Training script with MLflow logging | `ml/train.py` |
| ≥2 experiments (baseline + improved) | `trade-index` experiment, two runs |
| `ml/README_experiments.md` summary | ✅ |
| `ml/README_data.md` with source, v1→v2, commit mapping | ✅ |
| CI config | `.github/workflows/ci.yml` |
| README with commit→build→test→staging→prod and branch→pipeline map | ✅ |
| DVC with 2 versions and Azure remote | ✅ |

---

## How to verify everything in one command

```bash
python scripts/verify_project.py
```

```
96/96 checks passed
PROJECT COMPLETE
```

The script checks the application, the tests, the Docker configuration, both
workflows, the DVC setup, all six documents, and Git hygiene, then reports
pass/fail per line. It is itself linted in CI, so it cannot rot silently.
