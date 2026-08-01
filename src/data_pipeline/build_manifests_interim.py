# ruff: noqa
from __future__ import annotations
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any
import pandas as pd
from dotenv import load_dotenv

DATASET_CONFIGURE = {
    "small_dataset": {
        "id": "small-dataset",
        "worksheet": "breast-usg.xlsx",
        "required_columns": [
            "CaseID",
            "Image_filename",
            "Mask_tumor_filename",
            "Mask_other_filename",
            "Classification",
        ],
    },
    "medium_dataset": {"id": "medium-dataset", "worksheet": "medium-dataset-780"},
    "big_dataset": {"id": "big-dataset", "worksheet": "big-dataset-9.248"},
}
CLASS_TO_INDEX = {
    "benign": 0,
    "malignant": 1,
}
COLUMN_MAP = {
    "CaseID": "case_id",
    "Image_filename": "image_filename",
    "Mask_tumor_filename": "mask_tumor_filename",
    "Mask_other_filename": "mask_other_filename",
    "Classification": "classification",
}


def buid_interim_manifest(
    s_dataset_root: Path,
    s_output_root: Path,
    m_dataset_root: Path,
    m_output_root: Path,
    b_dataset_root: Path,
    b_output_root: Path,
):
    # Small Dataset
    s_img_metadata = s_dataset_root / DATASET_CONFIGURE["small_dataset"]["worksheet"]

    s_data = pd.read_excel(s_img_metadata, engine="openpyxl")
    s_data = s_data.rename(columns=COLUMN_MAP)
    s_samples: list[dict[str, Any]] = []
    s_classes: Counter[str] = Counter()
    s_samples_copy: set[str] = set()

    for _, row in s_data.iterrows():
        case = int(row["case_id"])
        case_id = f"case{case:04d}"
        img_filename = str(row["image_filename"])
        label = row["classification"]
        label_id = CLASS_TO_INDEX.get(label)
        s_classes[label] += 1

        s_samples_copy.add(img_filename)

        s_samples.append(
            {
                "sample_id": f"{DATASET_CONFIGURE['small_dataset']['id']}-{case:04d}",
                "case_id": case_id,
                "label_id": label_id,
                "label": label,
            }
        )

        manifest = {
            "dataset_id": DATASET_CONFIGURE["small_dataset"]["id"],
            "dataset_stage": "interim",
            "src": {
                "dataset_root_relative_path": ("src/data/raw/small-dataset"),
                "worksheet": DATASET_CONFIGURE["small_dataset"]["worksheet"],
            },
            "total_cases": len(s_samples),
            "samples": s_samples,
        }

        s_output_root.parent.mkdir(parents=True, exist_ok=True)

        with s_output_root.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)

    # medium dataset
    class_dirs = {
        "benign": m_dataset_root / "benign",
        "malignant": m_dataset_root / "malignant",
    }
    m_samples: list[dict[str, Any]] = []
    m_classes: Counter[str] = Counter()

    for label_m, class_dir in class_dirs.items():
        label_id_m = CLASS_TO_INDEX[label_m]
        img_files = sorted(f for f in class_dir.iterdir() if f.is_file())
        for case, img_path in enumerate(img_files, start=1):
            m_classes[label_m] += 1
            m_samples.append(
                {
                    "sample_id": f"{DATASET_CONFIGURE['medium_dataset']['id']}-{case:04d}",
                    "label_id": label_id_m,
                    "label": label_m,
                    "image_filename": img_path.name,
                }
            )

            manifest = {
                "dataset_id": DATASET_CONFIGURE["medium_dataset"]["id"],
                "dataset_stage": "interim",
                "src": {
                    "dataset_root_relative_path": ("src/data/raw/medium-dataset"),
                    "worksheet": DATASET_CONFIGURE["medium_dataset"]["worksheet"],
                },
                "total_cases": len(m_samples),
                "samples": m_samples,
            }

            m_output_root.parent.mkdir(parents=True, exist_ok=True)

            with m_output_root.open("w", encoding="utf-8") as file:
                json.dump(manifest, file, ensure_ascii=False, indent=2)

    # big dataset
    b_class_dirs = {
        "train": {
            "benign": b_dataset_root / "train" / "benign",
            "malignant": b_dataset_root / "train" / "malignant",
        },
        "val": {
            "benign": b_dataset_root / "val" / "benign",
            "malignant": b_dataset_root / "val" / "malignant",
        },
    }

    b_samples: list[dict[str, Any]] = []
    b_classes: Counter[str] = Counter()


if __name__ == "__main__":
    load_dotenv()
    project_root = Path(__file__).resolve().parents[2]

    s_root_from_env = os.getenv("SMALL_DATASET_ROOT")
    s_dataset_root = (
        Path(s_root_from_env)
        if s_root_from_env
        else project_root / "src" / "data" / "raw" / "small-dataset"
    )
    s_output_path = project_root / "src" / "data" / "interim" / "manifest_small.json"

    m_root_from_env = os.getenv("MEDIUM_DATASET_ROOT")
    m_dataset_root = (
        Path(m_root_from_env)
        if m_root_from_env
        else project_root / "src" / "data" / "raw" / "medium-dataset-780"
    )
    m_output_path = project_root / "src" / "data" / "interim" / "manifest_medium.json"

    # buid_interim_manifest(s_dataset_root, s_output_path, m_dataset_root, m_output_path)
