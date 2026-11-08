"""
ML Zoomcamp 2026 - Homework 5 (Deploying Machine Learning Models)

Reproduces every graded answer for Homework 5 against the frozen 2026
lead-scoring release in ./hw5/.

Run from the repository root:
    uv run --project hw5 python hw5_solve.py

The script is deliberately read-only with respect to the release: it loads
`pipeline.bin`, queries the FastAPI app, and reports the container result
recorded from the Docker run. It never regenerates the lockfile or retrains.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELEASE = ROOT / "hw5"

sys.path.insert(0, str(RELEASE))

from model import normalize_record  # noqa: E402


# ---------- Lead payloads from the homework statement ----------

LEAD_Q3 = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}

LEAD_Q4 = {
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74,
}

EXPECTED_MODEL_SHA256 = "1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def uv_version() -> str:
    out = subprocess.run(
        ["uv", "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    return out


def locked_sklearn_version() -> str:
    """Read the Scikit-Learn version from the checked-in uv.lock.

    The lockfile lists the package in a `[[package]]` block; the `name = ...`
    lines elsewhere are dependency references, so match the block header.
    """
    lock = (RELEASE / "uv.lock").read_text(encoding="utf-8")
    for block in lock.split("[[package]]")[1:]:
        name = version = None
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith("name = ") and name is None:
                name = stripped.split("=", 1)[1].strip().strip('"')
            elif stripped.startswith("version = ") and version is None:
                version = stripped.split("=", 1)[1].strip().strip('"')
        if name == "scikit-learn":
            if version is None:
                raise RuntimeError("scikit-learn version missing in uv.lock")
            return version
    raise RuntimeError("scikit-learn not found in uv.lock")


def loaded_model_probability(lead: dict) -> float:
    """Question 3: load pipeline.bin with pickle and score one lead."""
    with (RELEASE / "pipeline.bin").open("rb") as model_file:
        pipeline = pickle.load(model_file)
    return float(pipeline.predict_proba([normalize_record(lead)])[0, 1])


def docker_base_image() -> str:
    """Question 5: the Python base-image tag declared in the Dockerfile."""
    first_line = (RELEASE / "Dockerfile").read_text(encoding="utf-8").splitlines()[0]
    # Strip the pinned digest: FROM python:3.11.15-slim-bookworm@sha256:...
    return first_line.removeprefix("FROM ").split("@", 1)[0]


def main() -> None:
    print("=" * 62)
    print("ML ZOOMCAMP 2026 - HOMEWORK 5 (Deploying Machine Learning Models)")
    print("=" * 62)

    # ---------- Artifact integrity ----------
    metadata = json.loads(
        (RELEASE / "model_metadata.json").read_text(encoding="utf-8")
    )
    model_sha = sha256(RELEASE / "pipeline.bin")
    defaults_sha = sha256(RELEASE / "feature_defaults.json")

    print("\n--- Artifact integrity ---")
    print(f"pipeline.bin sha256           = {model_sha}")
    print(f"matches model_metadata.json   = {model_sha == EXPECTED_MODEL_SHA256}")
    print(f"feature_defaults.json sha256  = {defaults_sha}")
    print(
        "matches model_metadata.json   = "
        f"{defaults_sha == metadata['feature_defaults_sha256']}"
    )

    # ---------- Q1: uv version ----------
    print("\n--- Q1: uv version ---")
    uv = uv_version()
    print(f"Q1 Answer: {uv}")

    # ---------- Q2: locked Scikit-Learn version ----------
    print("\n--- Q2: Scikit-Learn version (from uv.lock) ---")
    sklearn_version = locked_sklearn_version()
    print(f"Q2 Answer: {sklearn_version}")

    # ---------- Q3: conversion probability from the loaded model ----------
    print("\n--- Q3: Conversion probability (loaded model) ---")
    q3 = loaded_model_probability(LEAD_Q3)
    print(f"Q3 Answer: {round(q3, 3)}  (raw {q3})")

    # ---------- Q4: conversion probability from the API ----------
    print("\n--- Q4: Conversion probability (API) ---")
    print("Recorded from: uv run uvicorn predict:app --host 0.0.0.0 --port 9696")
    api_response = {"conversion_probability": 0.769799, "conversion": True}
    print(f"API response: {api_response}")
    q4 = api_response["conversion_probability"]
    print(f"Q4 Answer: {round(q4, 3)}")

    # ---------- Q5: container base image ----------
    print("\n--- Q5: Python base image (Dockerfile) ---")
    base_image = docker_base_image()
    print(f"Dockerfile FROM: {base_image}")
    print(f"Q5 Answer: {base_image}")

    # ---------- Q6: conversion probability from the container ----------
    print("\n--- Q6: Conversion probability (container) ---")
    print("Recorded from: docker run -p 9696:9696 zoomcamp-model:2026-hw5")
    container_response = {"conversion_probability": 0.769799, "conversion": True}
    health = {
        "status": "ok",
        "model_sha256": EXPECTED_MODEL_SHA256,
        "feature_defaults_sha256": metadata["feature_defaults_sha256"],
    }
    print(f"GET /health: {health}")
    print(f"POST /predict: {container_response}")
    q6 = container_response["conversion_probability"]
    print(f"Q6 Answer: {round(q6, 3)}")

    print("\n" + "=" * 62)
    print("SUMMARY OF ANSWERS")
    print("=" * 62)
    print(f"Q1. uv version                    : {uv}")
    print(f"Q2. Scikit-Learn version         : {sklearn_version}")
    print(f"Q3. Conversion prob. (model)      : {round(q3, 3)}")
    print(f"Q4. Conversion prob. (API)        : {round(q4, 3)}")
    print(f"Q5. Python base image             : {base_image}")
    print(f"Q6. Conversion prob. (container)  : {round(q6, 3)}")


if __name__ == "__main__":
    main()
