# Data documentation — NZ Overseas Trade Indexes

This document records where the training data comes from, how its two versions
differ, and which Git commit corresponds to each version. It answers the
question the project brief asks for: **"which exact dataset version produced
this model?"**

---

## 1. Source

| | |
|---|---|
| Dataset | New Zealand **Overseas Trade Indexes** (OTV / OTP series) |
| Publisher | Stats NZ (Tatauranga Aotearoa) |
| Original file | `raw_dataset.csv` |
| Raw size | 23,324,132 bytes (~23 MB) |
| Raw shape | 104,613 rows × 13 columns |
| Coverage | Period `1925.06` → `2024.06` |

The raw CSV ships with the course materials. It is **not** committed to Git;
DVC tracks it and stores the payload in Azure Blob Storage.

### Raw columns

| Column | Type | Notes |
|---|---|---|
| `Series_reference` | id | 1,159 distinct series identifiers |
| `Period` | text | `YYYY.MM`, quarterly (`03`, `06`, `09`, `12`) |
| `Data_value` | numeric | **the prediction target** |
| `STATUS` | category | FINAL / REVISED / CONFIDENTIAL / PROVISIONAL |
| `UNITS` | category | `Index`, `Dollars`, `Percent`, plus a lowercase `index` variant |
| `MAGNTUDE` | category | `0` / `3` — misspelled in the source data |
| `Subject` | category | 2 values (OTV volumes/values, OTP prices) |
| `Group` | category | 31 values |
| `Series_title_1` | category | 14 values |
| `Series_title_2` | category | 181 values, 2,118 empty |
| `Series_title_3` | category | 5 values, 2,859 empty |
| `Series_title_4` | category | 4 values, 86,228 empty |
| `Series_title_5` | category | 1 value, 104,339 empty (effectively useless) |

---

## 2. Version 1 — raw, as published

**Purpose:** the initial usable dataset, establishing a baseline that can be
trained on without any transformation beyond what pandas does when reading CSV.

| | |
|---|---|
| File | `data/raw/raw_dataset.csv` |
| DVC pointer | `data/raw/raw_dataset.csv.dvc` |
| DVC md5 | `42171ec985af1d3aed38f4d6f497030b` |
| Size | 23,324,132 bytes |
| Rows × columns | 104,613 × 13 |
| Git tag | `data-v1` |
| Commit | `a3b8237b84d580d1966ff3e041dab4589a9a3c6a` |

### Why v1 is not directly trainable

`Data_value` mixes three incomparable scales in one column:

| `UNITS` | Rows | Mean | Max |
|---|---|---|---|
| `Dollars` | 24,498 | 3,098,887 | 1,123,282,000 |
| `Index` | 77,337 | 1,055 | 13,601 |
| `Percent` | 1,877 | 1.75 | 44.2 |

A single regression model trained on the mixture is dominated by the dollar
rows: the loss is driven almost entirely by values six orders of magnitude
larger than the rest. v2 exists to fix exactly this.

---

## 3. Version 2 — cleaned and filtered

**Purpose:** a homogeneous, model-ready dataset with a single comparable target
scale, plus time features extracted from the period key.

| | |
|---|---|
| File | `data/processed/clean_v2.csv` |
| DVC pointer | `data/processed/clean_v2.csv.dvc` |
| DVC md5 | `a0f682da4b534942450d8f30ba6cca1d` |
| Size | 15,863,981 bytes |
| Rows × columns | 77,337 × 11 |
| Git tag | `data-v2` |
| Commit | `a5eafb02e76bf44c5541889097fcdef2008401e7` |

Built by:

```bash
python -m ml.data_pipeline \
  --input data/raw/raw_dataset.csv \
  --output data/processed/clean_v2.csv
```

### Every change from v1 to v2

| # | Change | Rows affected | Why |
|---|---|---|---|
| 1 | Renamed `MAGNTUDE` → `MAGNITUDE` | — | typo in the source data; a misspelled feature name is a bug waiting to happen |
| 2 | Dropped `Series_reference` | — | it is a series identifier; keeping it lets the model memorise individual series instead of learning a general relationship (**target leakage**) |
| 3 | Dropped `Series_title_5` | — | only 274 of 104,613 rows are non-empty; the column carries no usable signal |
| 4 | Normalised `UNITS` casing | 1,490 (`index` → `Index`) | the source mixes `Index` and `index`, which would otherwise become two different one-hot categories |
| 5 | Coerced `Data_value` to numeric, dropped null targets | **−901** | a supervised model cannot learn from a row with no label |
| 6 | Filled blank categoricals with `Unknown` | 2,859 in `Series_title_3`, 2,118 in `Series_title_2`, 86,228 in `Series_title_4` | keeps the row and encodes "not recorded" as its own category instead of discarding data |
| 7 | **Filtered to `UNITS == "Index"`** | **−26,375** | the primary fix: makes the target scale homogeneous (see §2) |
| 8 | Split `Period` into `year` and `month` | — | converts `1971.06` text into two numeric features the model can use |
| 9 | Dropped `MAGNITUDE` | — | after step 7 it collapses to a single value, so it would only add a useless one-hot column |
| 10 | Reordered columns, wrote CSV | — | stable, readable output |

**Row accounting:**

```
104,613  raw rows
   −901  no Data_value
= 103,712
 −26,375  not UNITS == "Index"
=  77,337  v2 rows
```

### v2 column list (11)

```
Period, Data_value, year, month, STATUS, Subject, Group,
Series_title_1, Series_title_2, Series_title_3, Series_title_4
```

### v2 feature list (10)

Numeric: `year`, `month`
Categorical: `STATUS`, `Subject`, `Group`, `Series_title_1`, `Series_title_2`,
`Series_title_3`, `Series_title_4`

Target: `Data_value`

---

## 4. Commit ↔ dataset version mapping

This is the traceability table the brief asks for.

| Dataset version | Git tag | Commit | DVC md5 | Rows | Raw size |
|---|---|---|---|---|---|
| v1 (raw) | `data-v1` | `a3b8237b84d580d1966ff3e041dab4589a9a3c6a` | `42171ec985af1d3aed38f4d6f497030b` | 104,613 | 23,324,132 B |
| v2 (cleaned) | `data-v2` | `a5eafb02e76bf44c5541889097fcdef2008401e7` | `a0f682da4b534942450d8f30ba6cca1d` | 77,337 | 15,863,981 B |

**Models in MLflow record which of these they used** via the `data_version` and
`data_dvc_hash` parameters on each run. See `ml/README_experiments.md`.

---

## 5. Reproducing a dataset version

`git switch` changes which dataset version is *referenced*; `dvc pull`
materialises the data. Both steps are needed.

```bash
# 1. Check what is currently materialised
dvc status

# 2. Go back to v1
git switch --detach data-v1
dvc pull
head -3 data/raw/raw_dataset.csv

# 3. Go forward to v2
git switch --detach data-v2
dvc pull
head -3 data/processed/clean_v2.csv

# 4. Return to your branch
git switch dev
dvc pull
```

> ⚠️ Switching versions **overwrites the working-copy data files**. Commit or
> discard local edits first.

---

## 6. Remote storage

| Setting | Value |
|---|---|
| DVC remote name | `myazure` |
| URL | `azure://dvc` |
| Storage account | `wut2026` |
| Authentication | SAS token, stored **only** in `.dvc/config.local` |

`.dvc/config.local` is git-ignored. **Never commit it** — it contains a
credential. Verify with:

```bash
git check-ignore -v .dvc/config.local
git ls-files | grep config.local   # must print nothing
```

---

## 7. Rules this repository follows

1. **Raw data never enters Git.** Only `.dvc` pointer files are committed.
2. **Every dataset change goes through `dvc add` + `git commit` + `git tag`.**
3. **A model must record the `data_dvc_hash` it was trained on** so the exact
   dataset can be recovered later.
4. **Credentials stay out of Git.** `.env`, `.dvc/config.local` are ignored.

Verify rule 1 at any time:

```bash
git ls-files | grep -E '\.(csv|parquet)$'   # must print nothing
```
