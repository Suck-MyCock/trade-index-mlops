# Current production model

What is deployed, what it was trained on, and how to reproduce it.

---

## Model

| | |
|---|---|
| Estimator | `RandomForestRegressor` |
| Container | scikit-learn `Pipeline` (one-hot encoder → forest) |
| Artifact | `models/model.joblib` (joblib, `compress=3`, ~15 MB) |
| Exported by | `python -m ml.train` (the `tuned-forest` run) |
| MLflow run name | `tuned-forest` |
| MLflow experiment | `trade-index` |

## Training provenance

| | |
|---|---|
| Dataset version | **v2** (cleaned, `UNITS == "Index"` only) |
| DVC md5 | `a0f682da4b534942450d8f30ba6cca1d` |
| Git tag for the data | `data-v2` |
| Rows | 77,337 (61,869 train / 15,468 test) |
| Split | 80/20 hold-out, `random_state=42` |
| Features | `year`, `month`, `STATUS`, `Subject`, `Group`, `Series_title_1..4` |
| Target | `Data_value` |
| Model config | `ml/configs/baseline.yaml` |

## Held-out performance

| Metric | Value |
|---|---|
| RMSE | **240.0274** |
| MAE | **154.4691** |
| R² | **0.9085** |

Baseline for comparison — `LinearRegression` on the same features and split:
RMSE 578.1820, MAE 324.5331, R² 0.4689.

## How this model was chosen

`tuned-forest` halves the error and nearly doubles R² relative to the linear
baseline, on an untouched hold-out split. The reasoning, plus the known
limitations, is in [README_experiments.md](../README_experiments.md).

## Reproducing the artifact

```bash
# 1. get the exact dataset
git switch --detach data-v2
dvc pull

# 2. retrain and export
python -m ml.train \
  --data data/processed/clean_v2.csv \
  --config ml/configs/baseline.yaml \
  --pointer data/processed/clean_v2.csv.dvc
```

Training is deterministic (`random_state=42`), so the metrics above reproduce
exactly.

## Verifying the deployed artifact

```bash
python -c "from app.model_loader import model_holder; model_holder.load(); \
print(model_holder.model_type, model_holder.data_version)"
# RandomForestRegressor v2
```

Or through the running service:

```bash
curl http://localhost:8001/health
```

`model_type` and `data_version` in that response are read from the loaded
artifact and its `run_summary.json`, so they describe what is actually serving,
not what this document claims.

## When to replace this model

Replace it when a new experiment beats R² 0.9085 on a comparable hold-out
**and** the improvement is explained, not just observed. Update this file, then
[PROVENANCE.md](../PROVENANCE.md), in the same commit.
