# Trade Index ML App

ML-powered application built with DevOps and MLOps practices.

## Table of contents

- [Branching and pipelines](#branching-and-pipelines)
- [CI pipeline](#ci-pipeline)
- [CD pipeline](#cd-pipeline)
- [Data versioning (DVC)](#data-versioning-dvc)
- [Experiment tracking (MLflow)](#experiment-tracking-mlflow)
- [Testing](#testing)
- [Project structure](#project-structure)

## Branching and pipelines

Merging path: `feature/*` -> `dev` -> `staging` -> `main`.

| Branch | Trigger | Pipeline |
|---|---|---|
| `feature/*` -> `dev` | pull request | CI: tests, lint, Docker build |
| `dev` -> `staging` | pull request | CI: tests, lint, Docker build |
| `staging` -> `main` | pull request | CI: tests, lint, Docker build |
| merge to `staging` | push | CD: build staging image |
| merge to `main` | push | CD: build production image |

## CI pipeline

<!-- TODO: phase 5 -->

## CD pipeline

<!-- TODO: phase 6 -->

## Data versioning (DVC)

<!-- TODO: phase 7 -->

## Experiment tracking (MLflow)

<!-- TODO: phase 7 -->

## Testing

<!-- TODO: phase 3 -->

## Project structure

<!-- TODO: final -->

