# Experiment documentation — trade index regression

This document records the MLflow experiments, compares them, and answers the two
questions the project brief asks for:

1. **Which run do we consider production-worthy, and why?**
2. **Which metric are we optimising for, and why does that metric matter for
   this application?**

---

## 1. Task definition

| | |
|---|---|
| Task | Supervised regression |
| Target | `Data_value` — the trade index value for one series/period |
| Dataset | `v2` (cleaned, Index units only) |
| DVC md5 | `a0f682da4b534942450d8f30ba6cca1d` |
| Rows | 77,337 (after split: 61,869 train / 15,468 test) |
| Split | 80/20 hold-out, `random_state=42` |
| Features | 2 numeric (`year`, `month`) + 7 categorical, one-hot encoded |

Categorical features are one-hot encoded with `handle_unknown="ignore"`, so a
category never seen during training produces an all-zero block at inference time
instead of raising. The preprocessing is embedded in the saved sklearn
`Pipeline`, which means the API cannot accidentally apply a different
transformation than the one used for training.

---

## 2. Experiments

Both runs share the same data, split and metric code. Only the estimator
differs.

### Experiment A — `baseline-linear`

| | |
|---|---|
| Model | `LinearRegression(fit_intercept=True)` |
| Purpose | establish the reference point: what does the simplest linear model achieve on these features? |
| Hyperparameters | none beyond sklearn defaults |

| Metric | Value |
|---|---|
| RMSE | **578.1820** |
| MAE | 324.5331 |
| R² | **0.4689** |

### Experiment B — `tuned-forest`

| | |
|---|---|
| Model | `RandomForestRegressor` |
| Purpose | capture non-linear interactions between period and category that a linear model cannot express |
| Hyperparameters | `n_estimators=120`, `max_depth=20`, `min_samples_leaf=2`, `random_state=42` |
| Config file | `ml/configs/baseline.yaml` |

| Metric | Value |
|---|---|
| RMSE | **240.0274** |
| MAE | 154.4691 |
| R² | **0.9085** |

---

## 3. Comparison

| | `baseline-linear` | `tuned-forest` | Change |
|---|---|---|---|
| RMSE | 578.18 | **240.03** | **−58.5 %** |
| MAE | 324.53 | **154.47** | −52.4 % |
| R² | 0.4689 | **0.9085** | **+0.44** |

The forest halves the error and explains roughly 91 % of the variance in the
hold-out set, versus 47 % for the linear model.

**Reading the gap:** the linear model can only add the contributions of
`year`, `month` and each one-hot category. It cannot represent "the index for
*BEEF AND VEAL* trends differently from *Total Services*", nor interactions
between quarter and commodity. The forest can, which is where the extra 44
points of R² come from.

---

## 4. Which run is production-worthy?

**`tuned-forest`.**

Reasons, in order of weight:

1. **It is materially better on the hold-out set** — not a marginal gain. RMSE
   more than halves (578 → 240) and R² nearly doubles (0.469 → 0.909).
2. **The improvement generalises.** Both models are evaluated on the same
   untouched 20 % split, so the gap is not the result of tuning on test data.
3. **Its failure mode is acceptable.** The remaining error concentrates on small
   subsets (see §6), which is a known limitation rather than a defect.
4. **The linear model is kept only as a reference.** It is useful to prove the
   forest adds real value, but it is not a candidate for deployment: an R² of
   0.47 is too weak to be useful to a user.

The baseline run is therefore retained for comparison and reproducibility, but
the model registered as the current production candidate is `tuned-forest`.

---

## 5. Which metric are we optimising for, and why?

**RMSE (root mean squared error) on `Data_value`.**

Why RMSE for this application:

- **The target is a continuous index value**, so classification metrics
  (accuracy, F1) do not apply.
- **The error must be reported in the same unit as the prediction.** RMSE is in
  index points, which a domain user can interpret: "off by ±240 index points on
  average for large errors" is meaningful; R² alone is not.
- **RMSE penalises large errors more heavily than MAE** (it squares them). For
  a trade index, being off by a huge amount on one series is worse than being
  slightly off on many, so the extra penalty is desirable.

**R² is reported as the secondary metric** because it answers a different
question that RMSE cannot: *how much of the variance does the model explain
relative to simply predicting the mean?* It is the number that makes the
baseline-vs-tuned comparison legible.

**MAE is reported third** because it is robust to outliers and tells us the
typical error size without letting extreme values dominate.

---

## 6. Known limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Only `UNITS == "Index"` rows are modelled | The model cannot predict dollar or percent series | Train a separate model per unit, or add unit as a feature with a normalised target |
| `Series_title_2` has 180 categories | One-hot creates ~180 sparse columns; rare categories get little signal | Group rare categories, or switch to target encoding |
| Period is quarterly, but `month` is modelled as an integer | The model does not know quarters are cyclical | Encode month as `sin`/`cos`, or use a time-series model |
| Random split mixes periods | A car from 2024 may sit in train while a related 2023 row is in test | Use a **time-based split** (train on earlier periods, test on later) |
| No feature importance analysis yet | We do not know *which* features drive the prediction | Add permutation importance to the training script |

The random split is the most important limitation to state honestly: for a
time-indexed dataset, a chronological split is the more defensible evaluation,
and it will likely produce a lower (more realistic) R² than the 0.909 above.

---

## 7. How to reproduce

```bash
# 1. Activate the environment
.venv\Scripts\Activate.ps1          # Windows

# 2. Build v2 (only needed if data/processed/clean_v2.csv is absent)
python -m ml.data_pipeline \
  --input data/raw/raw_dataset.csv \
  --output data/processed/clean_v2.csv

# 3. Train both experiments
python -m ml.train \
  --data data/processed/clean_v2.csv \
  --config ml/configs/baseline.yaml \
  --pointer data/processed/clean_v2.csv.dvc

# 4. Inspect the runs
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
# open http://127.0.0.1:5000
```

Expected output of step 3:

```text
git commit : <current HEAD sha>
data hash  : a0f682da4b534942450d8f30ba6cca1d
train/test : 61869/15468
[baseline-linear] rmse=578.1820  mae=324.5331  r2=0.4689
[tuned-forest]    rmse=240.0274  mae=154.4691  r2=0.9085
```

Results are deterministic (`random_state=42`), so re-running reproduces the
numbers above exactly.

---

## 8. What every run records

The project brief requires five things per run. All five are logged:

| Requirement | MLflow field | Source |
|---|---|---|
| Code version (git commit SHA) | `git_commit` param | `git rev-parse HEAD` |
| Dataset version (DVC hash) | `data_dvc_hash` param + `data_version` | parsed from the `.dvc` pointer |
| Hyperparameters | `model_type`, `n_estimators`, `max_depth`, `min_samples_leaf`, `random_state` | config + code |
| Metrics | `rmse`, `mae`, `r2` | hold-out evaluation |
| Artifacts | logged model (`model.skops`) | `mlflow.sklearn.log_model` |

Train/test row counts are logged as well, so the split can be verified rather
than assumed.
