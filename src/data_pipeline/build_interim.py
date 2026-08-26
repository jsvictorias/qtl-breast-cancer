import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BINARY_LABEL_MAP = {"benign": 0, "malignant": 1}

SMALL_FILENAME_PATTERN = re.compile(
    r"^(?P<label>benign|malignant|normal)\s\((?P<idx>\d+)\)"
    r"(?P<mask_suffix>_mask(?:_\d+)?)?\.png$",
    re.IGNORECASE,
)

MEDIUM_ID_PATTERN = re.compile(r"^bus_(?P<case>\d{4})-(?P<side>[lrs])$")
BIG_FILENAME_PATTERN = re.compile(r"^(?P<label>benign|malignant)\((?P<idx>\d+)\)\.png$")


def _dataset_root(env_var: str, default: Path) -> Path:
    override = os.getenv(env_var)
    return Path(override) if override else default


def _rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _write_manifest(output_path: Path, manifest: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(
        f"[{manifest['dataset_id']}] wrote {manifest['total_samples']} samples "
        f"-> {output_path.relative_to(PROJECT_ROOT).as_posix()} "
        f"(unmatched: {len(manifest['issues']['unmatched_files'])})"
    )


def build_small_manifest(dataset_root: Path) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    unmatched: list[str] = []

    for label in sorted(BINARY_LABEL_MAP):
        class_dir = dataset_root / label
        if not class_dir.exists():
            continue

        images: dict[int, Path] = {}
        masks: dict[int, list[Path]] = defaultdict(list)

        for path in sorted(class_dir.glob("*.png")):
            match = SMALL_FILENAME_PATTERN.match(path.name)
            if not match or match.group("label").lower() != label:
                unmatched.append(_rel(path))
                continue
            idx = int(match.group("idx"))
            if match.group("mask_suffix") is None:
                images[idx] = path
            else:
                masks[idx].append(path)

        for idx, image_path in sorted(images.items()):
            case_id = f"{label}-{idx}"
            samples.append(
                {
                    "sample_id": f"small-{label}-{idx:04d}",
                    "case_id": case_id,
                    "label": label,
                    "label_id": BINARY_LABEL_MAP[label],
                    "image_relative_path": _rel(image_path),
                    "mask_relative_paths": [
                        _rel(mask_path) for mask_path in sorted(masks.get(idx, []))
                    ],
                    "group_key": f"small::{case_id}",
                    "metadata": {},
                }
            )

    return {
        "dataset_id": "small",
        "dataset_name": "BUSI - Breast Ultrasound Images",
        "dataset_stage": "interim",
        "source": {
            "raw_relative_path": _rel(dataset_root),
            "reference": "Al-Dhabyani et al., Data in Brief (2020)",
        },
        "label_map": BINARY_LABEL_MAP,
        "total_samples": len(samples),
        "issues": {"unmatched_files": unmatched},
        "samples": samples,
    }


def build_medium_manifest(dataset_root: Path) -> dict[str, Any]:
    metadata = pd.read_csv(dataset_root / "bus_data.csv")

    fold_lookup: dict[str, dict[str, int]] = defaultdict(dict)
    for fold_file, fold_key in (
        ("5-fold-cv.csv", "official_fold_5cv"),
        ("10-fold-cv.csv", "official_fold_10cv"),
    ):
        fold_path = dataset_root / fold_file
        if not fold_path.exists():
            continue
        fold_df = pd.read_csv(fold_path)
        for _, row in fold_df.iterrows():
            fold_lookup[row["ID"]][fold_key] = int(row["kFold"])

    images_dir = dataset_root / "Images"
    masks_dir = dataset_root / "Masks"

    samples: list[dict[str, Any]] = []
    unmatched: list[str] = []

    for _, row in metadata.iterrows():
        sample_key = str(row["ID"])
        match = MEDIUM_ID_PATTERN.match(sample_key)
        if not match:
            unmatched.append(sample_key)
            continue

        case, side = match.group("case"), match.group("side")
        image_path = images_dir / f"{sample_key}.png"
        mask_path = masks_dir / f"mask_{case}-{side}.png"
        if not image_path.exists():
            unmatched.append(_rel(image_path))
            continue

        label = str(row["Pathology"]).strip().lower()
        samples.append(
            {
                "sample_id": f"medium-{sample_key}",
                "case_id": f"case-{case}",
                "label": label,
                "label_id": BINARY_LABEL_MAP.get(label),
                "image_relative_path": _rel(image_path),
                "mask_relative_paths": [_rel(mask_path)] if mask_path.exists() else [],
                "group_key": f"medium::case-{case}",
                "metadata": {
                    "histology": row["Histology"],
                    "birads": str(row["BIRADS"]),
                    "device": row["Device"],
                    "width": int(row["Width"]),
                    "height": int(row["Height"]),
                    "side": row["Side"],
                    "bbox": row["BBOX"],
                    **fold_lookup.get(sample_key, {}),
                },
            }
        )

    return {
        "dataset_id": "medium",
        "dataset_name": "BUS-BRA - Breast Ultrasound Dataset",
        "dataset_stage": "interim",
        "source": {
            "raw_relative_path": _rel(dataset_root),
            "reference": "Gomez-Flores et al., Medical Physics (2024)",
        },
        "label_map": BINARY_LABEL_MAP,
        "total_samples": len(samples),
        "issues": {"unmatched_files": unmatched},
        "samples": samples,
    }


def build_big_manifest(dataset_root: Path) -> dict[str, Any]:
    cohort_root = dataset_root / "GDPH&SYSUCC"
    birads_by_id = {
        str(row["ID"]): row
        for _, row in pd.read_excel(
            cohort_root / "BIRADS&FOLD.xlsx", sheet_name="prediction", engine="openpyxl"
        ).iterrows()
    }

    samples: list[dict[str, Any]] = []
    unmatched: list[str] = []

    for institution in ("GDPH", "SYSUCC"):
        institution_dir = cohort_root / institution
        for path in sorted(institution_dir.glob("*.png")):
            match = BIG_FILENAME_PATTERN.match(path.name)
            if not match:
                unmatched.append(_rel(path))
                continue

            label = match.group("label").lower()
            idx = int(match.group("idx"))
            sample_key = f"{label}({idx})"
            birads_row = birads_by_id.get(sample_key)

            samples.append(
                {
                    "sample_id": f"big-{institution}-{label}-{idx:04d}",
                    "case_id": sample_key,
                    "label": label,
                    "label_id": BINARY_LABEL_MAP[label],
                    "image_relative_path": _rel(path),
                    "mask_relative_paths": [],
                    "group_key": f"big::{institution}::{sample_key}",
                    "metadata": {
                        "institution": institution,
                        "birads_reader1": (
                            str(birads_row["BIRADS-reader1"])
                            if birads_row is not None
                            else None
                        ),
                        "birads_reader2": (
                            str(birads_row["BIRADS-reader2"])
                            if birads_row is not None
                            else None
                        ),
                        "official_fold": (
                            int(birads_row["fold"]) if birads_row is not None else None
                        ),
                    },
                }
            )

    return {
        "dataset_id": "big",
        "dataset_name": "GDPH&SYSUCC Breast Ultrasound Dataset",
        "dataset_stage": "interim",
        "source": {
            "raw_relative_path": _rel(dataset_root),
            "reference": "Zhang et al., Healthcare (2022)",
        },
        "label_map": BINARY_LABEL_MAP,
        "total_samples": len(samples),
        "issues": {"unmatched_files": unmatched},
        "samples": samples,
    }


def main() -> None:
    load_dotenv()

    small_root = _dataset_root(
        "SMALL_DATASET_ROOT", PROJECT_ROOT / "data" / "raw" / "small"
    )
    medium_root = _dataset_root(
        "MEDIUM_DATASET_ROOT", PROJECT_ROOT / "data" / "raw" / "medium"
    )
    big_root = _dataset_root("BIG_DATASET_ROOT", PROJECT_ROOT / "data" / "raw" / "big")

    interim_dir = PROJECT_ROOT / "data" / "interim"

    _write_manifest(
        interim_dir / "manifest_small.json", build_small_manifest(small_root)
    )
    _write_manifest(
        interim_dir / "manifest_medium.json", build_medium_manifest(medium_root)
    )
    _write_manifest(interim_dir / "manifest_big.json", build_big_manifest(big_root))


if __name__ == "__main__":
    main()
