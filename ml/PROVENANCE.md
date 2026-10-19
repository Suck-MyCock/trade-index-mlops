# Provenance

The chain from raw data to the serving artifact, with every link checkable.

This file answers one question: **for the model currently in production, which
data and which code produced it?**

---

## The chain

```text
Stats NZ Overseas Trade Indexes  (upstream publication)
        │
        ├─ v1  data/raw/raw_dataset.csv
        │      DVC md5  42171ec985af1d3aed38f4d6f497030b
        │      tag      data-v1   →  commit a3b8237
        │      104,613 rows × 13 columns
        │
        │  ml/data_pipeline.py  (rename, drop, filter, time features)
        ▼
        ├─ v2  data/processed/clean_v2.csv
        │      DVC md5  a0f682da4b534942450d8f30ba6cca1d
        │      tag      data-v2   →  commit a5eafb0
        │      77,337 rows × 11 columns
        │
        │  ml/train.py  (80/20 split, random_state=42)
        ▼
        ├─ experiment  trade-index / tuned-forest
        │      params   git_commit, data_version=v2, data_dvc_hash=a0f682da…
        │      metrics  rmse=240.0274  mae=154.4691  r2=0.9085
        │
        ▼
        └─ artifact  models/model.joblib  (joblib, compress=3, ~15 MB)
               │
               │  app/model_loader.py  (loaded once at startup)
               ▼
           FastAPI service  GET /health   POST /predict
               │
               │  .github/workflows/ci.yml + cd.yml
               ▼
           container image  trade-index:{staging,production}
```

---

## Link-by-link verification

### 1. Is the data actually this data?

```bash
git show data-v2 --stat                       # the pointer was committed here
cat data/processed/clean_v2.csv.dvc           # md5 + size
md5sum data/processed/clean_v2.csv            # matches the pointer (Linux)
```

### 2. Which commit corresponds to which dataset?

```bash
git tag --list                                # data-v1, data-v2
git log -1 --format='%H %s' data-v1
git log -1 --format='%H %s' data-v2
```

Recorded in [README_data.md](README_data.md) §4.

### 3. Which experiment used which dataset?

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
# open the tuned-forest run → Parameters → data_dvc_hash
```

Expect `a0f682da4b534942450d8f30ba6cca1d`.

### 4. Which artifact is deployed?

```bash
cat models/run_summary.json
```

```json
{
  "git_commit": "73de586a2402a12893b553c56d37ad7db90cfd6f",
  "data_version": "v2",
  "data_dvc_hash": "a0f682da4b534942450d8f30ba6cca1d",
  "model_type": "RandomForestRegressor"
}
```

### 5. Does the running service agree?

```bash
curl http://localhost:8001/health
```

`model_type` and `data_version` in the response come from the loaded artifact,
so they report reality rather than intent.

---

## Why the hash and not just the tag

A Git tag is a human label that can be moved. The DVC md5 is computed from the
file contents, so it changes the moment a single byte changes and cannot be
faked. Recording both means:

- a human can read `data-v2`,
- a machine can verify `a0f682da…`.

---

## Known gaps in this chain

| Gap | Impact | Fix |
|---|---|---|
| `run_summary.json` is written by `ml/train.py`, not derived from the MLflow run | a mismatch would not be detected automatically | read the values back from MLflow and assert equality before export |
| The `git_commit` recorded is the commit at training time, which is one commit behind the commit that finally stores the exported artifact | the recorded SHA identifies the code that trained, not the file that shipped | acceptable, but state it: the artifact is reproducible from that SHA |
| No registry stage transition | `current_model.md` is the source of truth, not MLflow Registry | promote the chosen run in the Registry and read the stage from there |
| The image tags are not pushed to a registry | the artifact exists only on the runner | add a push step (see [DEPLOYMENT.md](../DEPLOYMENT.md) §7) |

None of these affects the correctness of what is deployed; they are the next
increments of rigour.
