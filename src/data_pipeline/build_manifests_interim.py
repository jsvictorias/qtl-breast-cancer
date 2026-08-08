# ruff: noqa
from __future__ import annotations
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any
import pandas as pd
from dotenv import load_dotenv
import re

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
B_FILENAME_PATTERN = re.compile(
    r"^(?P<cls>benign|malignant)\s*\((?P<base_id>\d+)\)(?:\s*-\s*(?P<aug_chain>.+))?\.png$"
)


def b_pattern_filenames(filename: str) -> dict[str, Any] | None:
    match = B_FILENAME_PATTERN.match(filename)
    if not match:
        return None

    aug_chain_raw = match.group("aug_chain")
    augmentations = (
        [tag.strip() for tag in aug_chain_raw.split("-") if tag.strip()]
        if aug_chain_raw
        else []
    )

    return {
        "class_from_filename": match.group("cls").lower(),
        "base_id": int(match.group("base_id")),
        "augmentations": augmentations,
        "is_original": len(augmentations) == 0,
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

        mask_tumor_filename = str(row["mask_tumor_filename"])
        mask_other_filename = str(row["mask_other_filename"])

        s_samples.append(
            {
                "sample_id": f"{DATASET_CONFIGURE['small_dataset']['id']}-{case:04d}",
                "case_id": case_id,
                "label_id": label_id,
                "label": label,
                "image_filename": img_filename,
                "mask_tumor_filename": mask_tumor_filename,
                "mask_other_filename": mask_other_filename,
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
    m_case_counter = 0

    for label_m, class_dirs in class_dirs.items():
        label_id_m = CLASS_TO_INDEX[label_m]
        img_files = sorted(
            f for f in class_dirs.iterdir() if f.is_file() and "_mask" not in f.stem
        )
        for img_path in img_files:
            m_case_counter += 1
            m_classes[label_m] += 1
            m_samples.append(
                {
                    "sample_id": f"{DATASET_CONFIGURE['medium_dataset']['id']}-{m_case_counter:04d}",
                    "label_id": label_id_m,
                    "label": label_m,
                    "image_filename": img_path.name,
                }
            )

            manifest = {
                "dataset_id": DATASET_CONFIGURE["medium_dataset"]["id"],
                "dataset_stage": "interim",
                "src": {
                    "dataset_root_relative_path": ("src/data/raw/medium-dataset-780"),
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
    b_no_pattern: list[str] = []
    b_case_counter = 0

    for split, classes in b_class_dirs.items():
        for label, class_dir in classes.items():
            for img_path in sorted(class_dir.glob("*.png")):
                patterned = b_pattern_filenames(img_path.name)
                if patterned is None:
                    b_no_pattern.append(str(img_path))
                    continue
                b_case_counter += 1
                b_samples.append(
                    {
                        "sample_id": f"{DATASET_CONFIGURE['big_dataset']['id']}-{b_case_counter:04d}",
                        "filename": img_path.name,
                        "split": split,
                        "label": label,
                        "augmentations": patterned["augmentations"],
                        "is_original": patterned["is_original"],
                    }
                )
                manifest = {
                    "dataset_id": DATASET_CONFIGURE["big_dataset"]["id"],
                    "dataset_stage": "interim",
                    "src": {
                        "dataset_root_relative_path": (
                            "src/data/raw/big-dataset-9.248"
                        ),
                        "worksheet": DATASET_CONFIGURE["big_dataset"]["worksheet"],
                    },
                    "total_cases": len(b_samples),
                    "samples": b_samples,
                    "no_pattern": b_no_pattern,
                }

                b_output_root.parent.mkdir(parents=True, exist_ok=True)

                with b_output_root.open("w", encoding="utf-8") as file:
                    json.dump(manifest, file, ensure_ascii=False, indent=2)


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

    b_root_from_env = os.getenv("BIG_ROOT_DATASET")
    b_dataset_root = (
        Path(b_root_from_env)
        if b_root_from_env
        else project_root / "src" / "data" / "raw" / "big-dataset-9.248"
    )
    b_output_path = project_root / "src" / "data" / "interim" / "manifest_big.json"

    buid_interim_manifest(
        s_dataset_root,
        s_output_path,
        m_dataset_root,
        m_output_path,
        b_dataset_root,
        b_output_path,
    )
