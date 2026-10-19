# Deployment

How the staging and production images are built, what configuration each one
gets, and the trade-offs behind those choices.

> **Scope:** the brief asks for *at least one automated deployment stage* that
> builds an image. It explicitly does **not** ask for the image to be pushed to a
> registry or started on a server. This document covers image building only, and
> the section [Extending to a real deploy](#extending-to-a-real-deploy) sketches
> what a genuine rollout would add.

---

## 1. Pipeline shape

`.github/workflows/cd.yml` contains two independent jobs. Only one of them runs
for any given merge, because each is gated on a branch:

| Job | `if` condition | Environment | Image tags |
|---|---|---|---|
| `build-staging` | `github.ref == 'refs/heads/staging'` | `staging` | `trade-index:staging`, `trade-index:staging-<sha12>` |
| `build-production` | `github.ref == 'refs/heads/main'` | `production` | `trade-index:production`, `trade-index:production-<sha12>` |

Verified behaviour: merging into `staging` runs `build-staging` and reports
`build-production` as skipped; merging into `main` does the reverse.

---

## 2. Why GitHub Environments

A single workflow file serves both environments. The `environment:` key is what
makes them differ:

```yaml
jobs:
  build-staging:
    environment: staging          # secrets resolve from the staging environment
```

Consequences:

1. `secrets.APP_ENV` resolves to the **staging** value in one job and the
   **production** value in the other, with no duplicated workflow code.
2. Each deployment is recorded against its environment on the repository's
   Deployments page, so the two release tracks stay auditable and separate.
3. An environment can later require reviewers, a wait timer, or a branch
   restriction without touching the workflow.

### Environment secrets to create

`Settings → Environments → New environment`, then add:

| Environment | Secret | Suggested value |
|---|---|---|
| `staging` | `APP_ENV` | `staging` |
| `staging` | `LOG_LEVEL` | `INFO` |
| `production` | `APP_ENV` | `production` |
| `production` | `LOG_LEVEL` | `WARNING` |

`Settings → Branches` should then protect `dev`, `staging` and `main` and
require the `Test, lint and build` status check, so a red CI blocks the merge
that would otherwise trigger CD.

---

## 3. What each CD job does

```yaml
- name: Create staging .env from secrets
  env:
    APP_ENV: ${{ secrets.APP_ENV }}
    LOG_LEVEL: ${{ secrets.LOG_LEVEL }}
  run: |
    : "${APP_ENV:=staging}"
    : "${LOG_LEVEL:=INFO}"
    {
      echo "APP_ENV=${APP_ENV}"
      echo "LOG_LEVEL=${LOG_LEVEL}"
    } > .env
```

The `.env` is written into the runner's workspace. Its values are then sourced
and forwarded to the build:

```yaml
- name: Build staging image
  run: |
    set -a; . ./.env; set +a
    docker build \
      --build-arg APP_ENV="${APP_ENV}" \
      --build-arg LOG_LEVEL="${LOG_LEVEL}" \
      --tag "trade-index:staging" \
      --tag "trade-index:staging-${GITHUB_SHA::12}" \
      .
```

### Why build arguments instead of `COPY .env`

The brief describes creating a `.env` and then building the image. Copying that
file into the image would satisfy the letter of the instruction but bake
configuration — and, in a real project, credentials — into an immutable layer
where anyone with the image can read it.

Passing the values as `--build-arg` keeps the mechanism identical while avoiding
that: the Dockerfile declares

```dockerfile
ARG APP_ENV=local
ARG LOG_LEVEL=INFO
ENV APP_ENV=${APP_ENV} \
    LOG_LEVEL=${LOG_LEVEL}
```

so the running container reports the right environment, and no secret file is
stored in the image. `.dockerignore` excludes `.env` for the same reason.

**Trade-off, stated plainly:** build arguments are visible in
`docker history`. They are fine for `APP_ENV` and `LOG_LEVEL`, which are not
sensitive. A real database password must be injected at run time with
`docker run --env-file` or a secrets manager, never at build time.

---

## 4. Verifying a run

After merging into `staging`, open the `CD` run and inspect the
`.env` step. GitHub masks secret values, so the log shows:

```text
Generated .env:
APP_ENV=***
LOG_LEVEL=***
```

The `***` is the point: it proves the values came from the environment's secret
store and were populated, while keeping them out of the log. The image-build
step then shows two tags being created, one of them masked because the literal
string `staging` also happens to be a secret value.

To confirm the two environments really differ, compare the `.env` step in the
`staging` run with the one in the `main` run — both mask their values, but the
Deployments page (`/<owner>/<repo>/deployments`) lists them against different
environments.

---

## 5. Running the built image

```bash
docker build -t trade-index:local .
docker run --rm -p 127.0.0.1:8001:8000 -e APP_ENV=local trade-index:local
```

```bash
curl http://localhost:8001/health
# {"status":"ok","environment":"local","model_loaded":true,
#  "model_type":"RandomForestRegressor","data_version":"v2"}
```

`environment` echoes `APP_ENV`, which is how the CD pipeline's configuration
becomes visible at runtime.

---

## 6. Failure modes and what they mean

| Symptom | Likely cause | Check |
|---|---|---|
| CD job skipped on a merge | the merge landed on the wrong branch | the job's `if:` compares `github.ref` |
| `Create .env` shows empty values | environment secrets not created, or created at repository level instead of in the environment | `Settings → Environments` |
| `docker build` fails on `COPY models/` | `models/model.joblib` missing from the commit | `git ls-files models/` |
| Image runs but `/health` says `model_loaded: false` | the artifact was excluded by `.dockerignore` | `docker run … ls models/` |
| Merge button stays disabled | required status check name does not match the job name | the check must be exactly `Test, lint and build` |

---

## 7. Extending to a real deploy

The brief stops at image building. A genuine rollout would add:

1. **Push to a registry** — `docker/login-action` plus
   `docker/build-push-action` with `push: true`, tagging
   `<registry>/<owner>/trade-index:{staging,production}-<sha>`.
2. **A deploy step** — pull that tag on the target host and restart the
   container, or roll it out through Kubernetes/ECS.
3. **A health gate** — after restart, poll `/health` until `model_loaded` is
   true, and roll back on failure.
4. **Runtime secrets** — inject database credentials with `--env-file` or a
   managed secret store rather than build arguments.
5. **Model provenance at deploy time** — assert that the image's
   `run_summary.json` names the intended `data_dvc_hash` before promoting it.
