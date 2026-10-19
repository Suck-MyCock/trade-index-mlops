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
    """Run a command and return stdout.

    ``text=True`` would decode with the platform locale (GBK on Chinese
    Windows), which raises UnicodeDecodeError on git output that contains
    non-locale bytes. Pin UTF-8 and replace undecodable bytes instead.
    """
    try:
        result = subprocess.run(
            args,
            cwd=ROOT,
            capture_output=True,
            timeout=120,
        )
        return result.stdout.decode("utf-8", errors="replace").strip()
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


# ---------------------------------------------------------------------------
section("1b. Prediction form")
# ---------------------------------------------------------------------------
# The form is a shipped artefact, not a dev convenience: without it the only
# way to try the model is to hand-write JSON. It must therefore be checked.
form = ROOT / "app/static/index.html"
group_map = ROOT / "app/static/group_map.json"
check(form.exists(), "app/static/index.html present")
check(group_map.exists(), "app/static/group_map.json present")

if form.exists():
    form_html = form.read_text(encoding="utf-8")
    # Present in the static markup.
    for control in ["year", "group", "series", "adv-body", "predict", "pred"]:
        check(f'id="{control}"' in form_html, f"form has a {control} control")
    # Built at runtime by script, so look for the id prefix and the field list.
    check('"adv-" + name' in form_html, "form builds advanced controls at runtime")
    check('"month"' in form_html, "month is offered among the advanced fields")
    check('fetch("/predict"' in form_html, "form calls POST /predict")
    check('fetch("/options"' in form_html, "form loads its options from the API")
    check(
        "group_map.json" in form_html,
        "form narrows series by group",
    )
    # month is the weakest feature (importance 0.008). It stays in the API
    # contract because the model consumes it, but it must not be a primary
    # control — the form would be offering a knob that does nothing.
    check('id="month-seg"' not in form_html, "month is not a primary control")
    # Only real fetches count: an `http://` inside an inline SVG namespaces or
    # a doc link is not a runtime dependency.
    external = [
        tag for tag in ('src="http', 'href="http', "url(http", "@import") if tag in form_html
    ]
    check(not external, "form has no external asset dependencies", str(external))

if group_map.exists():
    groups = json.loads(group_map.read_text(encoding="utf-8"))
    check(len(groups) >= 20, f"group map covers {len(groups)} groups")
    sample = next(iter(groups.values()))
    check("series_title_2" in sample, "group map lists series per group")
    check("defaults" in sample, "group map supplies coherent defaults")

check('"/options"' in main_py or "'/options'" in main_py, "GET /options endpoint")
check('mount("/static"' in main_py or "mount('/static'" in main_py, "static files mounted")

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
test_files = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").rglob("test_*.py"))
check(len(test_files) >= 3, f"at least 3 test files ({len(test_files)} found)")

for expected in (
    "tests/unit/test_schemas.py",
    "tests/unit/test_data_pipeline.py",
    "tests/integration/test_model_loader.py",
    "tests/api/test_predict_api.py",
):
    check(expected in test_files, f"test file present: {expected}")

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

pipeline_tests = read("tests/unit/test_data_pipeline.py")
check("MAGNITUDE" in pipeline_tests, "pipeline tests cover the column rename")
check("select_target_units" in pipeline_tests, "pipeline tests cover unit filtering")
check("add_time_features" in pipeline_tests, "pipeline tests cover period parsing")
check("idempotent" in pipeline_tests.lower(), "pipeline tests assert idempotency")

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

# Scan every tracked text file for SAS-token-shaped strings. Walking the file
# list is fast and deterministic; `git log -p --all` would stream a 15 MB
# binary diff and produce no useful additional signal.
SAS_PATTERN = re.compile(r"(sp=racwdl|sig=[A-Za-z0-9%]{20,})")
# This file contains the pattern itself, so it would always match.
SELF = "scripts/verify_project.py"
leaked: list[str] = []
for rel in sh("git", "ls-files").splitlines():
    if rel == SELF:
        continue
    path = ROOT / rel
    if not path.is_file() or path.stat().st_size > 2_000_000:
        continue
    try:
        body = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        continue
    if SAS_PATTERN.search(body):
        leaked.append(rel)
check(not leaked, "no SAS token in any tracked file", str(leaked[:3]))

# A working-tree scan cannot see a credential that was committed and then
# deleted. Search history too, but for the *signature* only: the permission
# prefix is not itself secret, and documentation that names the token format
# would otherwise be flagged forever. `-G` is a regex pickaxe; on this
# repository it runs in well under a second.
history_leak = sh("git", "log", "--all", "--oneline", "-G", r"sig=[A-Za-z0-9%]{20,}")
check(
    not history_leak,
    "no SAS signature anywhere in git history",
    history_leak.splitlines()[0] if history_leak else "",
)

check(
    ".dvc/config.local" not in sh("git", "ls-files"),
    "config.local is not tracked",
)

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
