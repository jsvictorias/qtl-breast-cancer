import json
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.json"


@pytest.fixture(scope="session")
def processed_manifest() -> dict[str, Any]:
    if not PROCESSED_MANIFEST_PATH.exists():
        pytest.skip(
            f"{PROCESSED_MANIFEST_PATH} not found — run "
            "`python -m src.data_pipeline.build_processed` first."
        )
    with PROCESSED_MANIFEST_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="session")
def samples(processed_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return processed_manifest["samples"]
