from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

DATASET_ID = "small-dataset"
WORKSHEET = "BrEaST-Lesions-USG clinical dat"

CLASS_TO_INDEX = {
    "benign": 0,
    "malignant": 1,
}

REQUIRED_COLUMNS = {
    "CaseID",
    "Image_filename",
    "Mask_tumor_filename",
    "Mask_other_filename",
    "Classification",
}

COLUMN_NAMES = {
    "CaseID": "case_id",
    "Image_filename": "image_filename",
    "Mask_tumor_filename": "mask_tumor_filename",
    "Mask_other_filename": "mask_other_filename",
    "Pixel_size": "pixel_size_cm",
    "Age": "age",
    "Tissue_composition": "tissue_composition",
    "Signs": "signs",
    "Symptoms": "symptoms",
    "Shape": "shape",
    "Margin": "margin",
    "Echogenicity": "echogenicity",
    "Posterior_features": "posterior_features",
    "Halo": "halo",
    "Calcifications": "calcifications",
    "Skin_thickening": "skin_thickening",
    "Interpretation": "interpretation",
    "BIRADS": "birads",
    "Verification": "verification",
    "Diagnosis": "diagnosis",
    "Classification": "classification",
}


def json_safe(value: Any) -> Any:
    """Converte valores do pandas/NumPy para tipos serializáveis em JSON."""
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def normalize_label(value: Any) -> str:
    """Normaliza a classe original sem descartar a classe normal."""
    label = str(value).strip().lower()

    aliases = {
        "benign": "benign",
        "benigno": "benign",
        "malignant": "malignant",
        "malign": "malignant",
        "maligno": "malignant",
        "normal": "normal",
    }

    if label not in aliases:
        raise ValueError(f"Classe desconhecida no XLSX: {value!r}")

    return aliases[label]


def create_sample_id(
    case_id: str,
    image_filename: str,
) -> str:
    """Cria um identificador estável para a amostra."""
    source = f"{DATASET_ID}:{case_id}:{image_filename}"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"small-{case_id}-{digest}"


def relative_file_path(
    images_directory: Path,
    filename: Any,
) -> str | None:
    """Retorna o caminho relativo esperado ou None quando não há arquivo."""
    if filename is None or pd.isna(filename):
        return None

    return (images_directory.name + "/" + str(filename)).replace("\\", "/")


def build_interim_manifest(
    dataset_root: Path,
    output_path: Path,
) -> None:
    images_directory = dataset_root / "img"
    metadata_path = dataset_root / "breast-usg.xlsx"

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Diretório do dataset não encontrado: {dataset_root}")

    if not images_directory.is_dir():
        raise FileNotFoundError(
            f"Diretório de imagens não encontrado: {images_directory}"
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Arquivo de metadados não encontrado: {metadata_path}")

    dataframe = pd.read_excel(
        metadata_path,
        sheet_name=WORKSHEET,
        engine="openpyxl",
    )

    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Colunas obrigatórias ausentes no XLSX: {sorted(missing_columns)}"
        )

    dataframe = dataframe.rename(columns=COLUMN_NAMES)

    samples: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()

    missing_primary_images: list[str] = []
    missing_tumor_masks: list[str] = []
    missing_other_masks: list[str] = []

    referenced_filenames: set[str] = set()

    for _, row in dataframe.iterrows():
        case_number = int(row["case_id"])
        case_id = f"case{case_number:03d}"

        image_filename = str(row["image_filename"])
        tumor_mask_filename = json_safe(row["mask_tumor_filename"])
        other_mask_filename = json_safe(row["mask_other_filename"])

        label = normalize_label(row["classification"])
        label_id = CLASS_TO_INDEX.get(label)
        eligible = label in CLASS_TO_INDEX

        class_counts[label] += 1

        primary_path = images_directory / image_filename

        if not primary_path.is_file():
            missing_primary_images.append(case_id)

        referenced_filenames.add(image_filename)

        if tumor_mask_filename is not None:
            tumor_mask_filename = str(tumor_mask_filename)
            referenced_filenames.add(tumor_mask_filename)

            if not (images_directory / tumor_mask_filename).is_file():
                missing_tumor_masks.append(case_id)

        if other_mask_filename is not None:
            other_mask_filename = str(other_mask_filename)
            referenced_filenames.add(other_mask_filename)

            if not (images_directory / other_mask_filename).is_file():
                missing_other_masks.append(case_id)

        clinical_metadata = {
            key: json_safe(value)
            for key, value in row.to_dict().items()
            if key
            not in {
                "case_id",
                "image_filename",
                "mask_tumor_filename",
                "mask_other_filename",
                "classification",
            }
        }

        samples.append(
            {
                "sample_id": create_sample_id(
                    case_id=case_id,
                    image_filename=image_filename,
                ),
                "case_id": case_id,
                "split": "unassigned",
                "label_original": label,
                "label": label,
                "label_id": label_id,
                "eligible_for_binary_task": eligible,
                "files": {
                    "image": relative_file_path(
                        images_directory,
                        image_filename,
                    ),
                    "tumor_mask": relative_file_path(
                        images_directory,
                        tumor_mask_filename,
                    ),
                    "other_mask": relative_file_path(
                        images_directory,
                        other_mask_filename,
                    ),
                },
                "clinical_metadata": clinical_metadata,
            }
        )

    physical_images = {
        path.name for path in images_directory.iterdir() if path.is_file()
    }

    unreferenced_files = sorted(physical_images.difference(referenced_filenames))

    manifest = {
        "schema_version": "1.0.0",
        "dataset_id": DATASET_ID,
        "dataset_stage": "interim",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "dataset_root_relative_path": ("src/data/raw/small-dataset"),
            "images_directory": "img",
            "metadata_file": "breast-usg.xlsx",
            "worksheet": WORKSHEET,
            "file_existence_validation": "performed",
        },
        "task_definition": {
            "final_task": "binary_image_classification",
            "target_classes": ["benign", "malignant"],
            "class_to_index": CLASS_TO_INDEX,
            "interim_labels_preserved": [
                "benign",
                "malignant",
                "normal",
            ],
            "normal_policy": ("keep_in_interim_and_exclude_from_processed"),
        },
        "statistics": {
            "total_cases": len(samples),
            "by_original_class": dict(sorted(class_counts.items())),
            "eligible_for_binary_task": sum(
                sample["eligible_for_binary_task"] for sample in samples
            ),
            "excluded_from_binary_task": sum(
                not sample["eligible_for_binary_task"] for sample in samples
            ),
            "primary_images_listed": sum(
                sample["files"]["image"] is not None for sample in samples
            ),
            "tumor_masks_listed": sum(
                sample["files"]["tumor_mask"] is not None for sample in samples
            ),
            "missing_primary_images": missing_primary_images,
            "missing_tumor_masks": missing_tumor_masks,
            "missing_other_masks": missing_other_masks,
            "unreferenced_files": unreferenced_files,
        },
        "samples": samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Manifest interim criado em: {output_path}")
    print(
        json.dumps(
            manifest["statistics"],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    load_dotenv()
    project_root = Path(__file__).resolve().parents[4]

    root_from_env = os.getenv("SMALL_DATASET_ROOT")

    dataset_root = (
        Path(root_from_env)
        if root_from_env
        else (project_root / "src" / "data" / "raw" / "small-dataset")
    )

    output_path = (
        project_root / "src" / "data" / "interim" / "manifests" / "small-dataset.json"
    )

    build_interim_manifest(
        dataset_root=dataset_root,
        output_path=output_path,
    )
