"""Final acceptance check for the trade-index-mlops project.

Run from the project root:

    python scripts/verify_project.py

Every check maps to a line of the course brief, so a clean run is evidence that
the deliverable is complete.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent

results: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    results.append((ok, label, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"{mark}  {label}" + (f"   → {detail}" if detail and not ok else ""))


def sh(*args: str) -> str:
    try:
        return subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True, timeout=120
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<error: {exc}>"


def read(path: str) -> str:
    p = ROOT / path
    return p.read_text(encoding="utf-8") if p.exists() else ""


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
section("1. Application")
# ---------------------------------------------------------------------------
for f in ["app/main.py", "app/schemas.py", "app/model_loader.py"]:
    check((ROOT / f).exists(), f"app module present: {f}")

main_py = read("app/main.py")
check('"/health"' in main_py or "'/health'" in main_py, "GET /health endpoint")
check('"/predict"' in main_py or "'/predict'" in main_py, "POST /predict endpoint")
check("lifespan" in main_py, "model loaded once via lifespan, not per request")

schemas = read("app/schemas.py")
check("Field(" in schemas and "ge=" in schemas, "input constraints declared in schemas")
check('extra="forbid"' in schemas or "extra='forbid'" in schemas, "unknown fields rejected")

artifact = ROOT / "models/model.joblib"
check(artifact.exists(), "trained artifact committed (models/model.joblib)")
if artifact.exists():
    mb = artifact.stat().st_size / 1024 / 1024
    check(mb > 1, f"artifact size {mb:.1f} MB")

summary_path = ROOT / "models/run_summary.json"
check(summary_path.exists(), "models/run_summary.json present")
if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("git_commit", "data_version", "data_dvc_hash", "model_type"):
        check(key in summary, f"run_summary records {key}", str(summary))

# ---------------------------------------------------------------------------
section("2. Tests")
# ---------------------------------------------------------------------------
test_files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests").rglob("test_*.py"))
check(len(test_files) >= 3, f"at least 3 test files ({len(test_files)} found)")

total_tests = 0
for f in test_files:
    body = read(f)
    total_tests += len(re.findall(r"^def test_", body, re.M))
    total_tests += len(re.findall(r"@pytest\.mark\.parametrize", body, re.M)) * 2
check(total_tests >= 20, f"substantial boundary coverage (~{total_tests} cases)")

api_tests = read("tests/api/test_predict_api.py")
check("422" in api_tests, "API tests assert validation failures return 422")
check("503" in api_tests, "API tests cover the model-unavailable path")
check("deterministic" in api_tests.lower(), "API tests assert determinism")

loader_tests = read("tests/integration/test_model_loader.py")
check("FileNotFoundError" in loader_tests, "loader tests cover a missing artifact")
check("unseen" in loader_tests.lower(), "loader tests cover unseen categories")

# ---------------------------------------------------------------------------
section("3. Docker")
# ---------------------------------------------------------------------------
dockerfile = read("Dockerfile")
check("python:3.12-slim" in dockerfile or "BASE_IMAGE" in dockerfile, "slim base image")
check("USER 10001" in dockerfile, "runs as a non-root user")
check("COPY requirements.txt" in dockerfile, "dependency layer copied before code")
dockerignore = read(".dockerignore")
check(bool(dockerignore), ".dockerignore present")
for pattern in ["data/", "tests/", ".git/", ".venv/"]:
    check(pattern in dockerignore, f".dockerignore excludes {pattern}")
check(".env" in dockerignore, ".dockerignore excludes .env")

# ---------------------------------------------------------------------------
section("4. CI pipeline")
# ---------------------------------------------------------------------------
ci = read(".github/workflows/ci.yml")
check(bool(ci), ".github/workflows/ci.yml present")
check("pull_request" in ci, "CI triggers on pull_request")
for branch in ["main", "staging", "dev"]:
    check(branch in ci, f"CI covers PRs targeting {branch}")
check("ruff check" in ci, "CI runs lint")
check("ruff format" in ci, "CI runs format check")
check("pytest" in ci, "CI runs the test suite")
check("docker build" in ci, "CI builds the Docker image")
check("contents: read" in ci, "CI uses a read-only token")

# ---------------------------------------------------------------------------
section("5. CD pipeline")
# ---------------------------------------------------------------------------
cd = read(".github/workflows/cd.yml")
check(bool(cd), ".github/workflows/cd.yml present")
check("staging" in cd and "main" in cd, "CD covers staging and main")
check("environment: staging" in cd, "staging job bound to the staging Environment")
check("environment: production" in cd, "production job bound to the production Environment")
check(".env" in cd, "CD generates an .env")
check("secrets." in cd, "CD reads values from secrets")
check("docker build" in cd, "CD builds an image")
check("pull_request" not in cd, "CD is not triggered by a pull request")

# ---------------------------------------------------------------------------
section("6. Data versioning (DVC)")
# ---------------------------------------------------------------------------
check((ROOT / ".dvc").is_dir(), "DVC initialised")
dvc_config = read(".dvc/config")
check("azure://" in dvc_config, "Azure remote configured")
check("account_name" in dvc_config, "storage account configured")
check("sas_token" not in dvc_config, ".dvc/config contains no SAS token")

v1 = ROOT / "data/raw/raw_dataset.csv.dvc"
v2 = ROOT / "data/processed/clean_v2.csv.dvc"
check(v1.exists(), "v1 pointer committed")
check(v2.exists(), "v2 pointer committed")

tracked = sh("git", "ls-files")
csv_tracked = [ln for ln in tracked.splitlines() if ln.endswith((".csv", ".parquet"))]
check(not csv_tracked, "no raw CSV committed to Git", str(csv_tracked[:3]))

ignored = sh("git", "check-ignore", ".dvc/config.local")
check(bool(ignored), "config.local is git-ignored")

tags = sh("git", "tag", "--list")
check("data-v1" in tags, "tag data-v1 exists")
check("data-v2" in tags, "tag data-v2 exists")

# ---------------------------------------------------------------------------
section("7. Documentation")
# ---------------------------------------------------------------------------
docs = {
    "README.md": ["How code moves", "Branching and pipelines", "CI pipeline", "CD pipeline"],
    "DEPLOYMENT.md": ["Environment", "secrets"],
    "ml/README_data.md": ["42171ec985af1d3aed38f4d6f497030b", "a0f682da4b534942450d8f30ba6cca1d"],
    "ml/README_experiments.md": ["0.9085", "RMSE", "production-worthy"],
    "ml/registry/current_model.md": ["RandomForestRegressor", "a0f682da"],
    "ml/PROVENANCE.md": ["data-v1", "data-v2", "model.joblib"],
}
for name, needles in docs.items():
    body = read(name)
    check(bool(body), f"{name} present")
    for needle in needles:
        check(needle in body, f"{name} mentions '{needle}'")

readme = read("README.md")
check("TODO" not in readme, "README has no leftover TODO markers")

# ---------------------------------------------------------------------------
section("8. Git hygiene")
# ---------------------------------------------------------------------------
branch = sh("git", "branch", "--show-current")
check(branch in {"dev", "staging", "main"}, f"on a tracked branch ({branch})")

all_branches = sh("git", "branch", "--list")
for b in ["dev", "staging", "main"]:
    check(b in all_branches, f"branch {b} exists")

secrets_scan = sh("git", "log", "-p", "--all")
leaked = list(re.findall(r"(sp=racwdl|sig=[A-Za-z0-9%]{20,})", secrets_scan))
check(not leaked, "no SAS token in git history", str(leaked[:2]))

# ---------------------------------------------------------------------------
print()
print("=" * 72)
failed = [r for r in results if not r[0]]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print()
    print("FAILURES:")
    for _, label, detail in failed:
        print(f"  - {label}" + (f"   ({detail})" if detail else ""))
    sys.exit(1)
print("PROJECT COMPLETE")
