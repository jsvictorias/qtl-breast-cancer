from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

DATASET_ID = "big-dataset"
RAW_DIRECTORY_NAME = "big-dataset-9.248"
EXPECTED_SPLITS = ("train", "val")
CLASS_TO_INDEX = {"benign": 0, "malignant": 1}
CLASS_ALIASES = {
    "benign": "benign",
    "bening": "benign",
    "benigno": "benign",
    "malignant": "malignant",
    "malign": "malignant",
    "maligno": "malignant",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def find_project_root(current_file: Path) -> Path:
    """Localiza a raiz do projeto procurando a pasta src/data."""
    for parent in current_file.resolve().parents:
        if (parent / "src" / "data").is_dir():
            return parent
    raise RuntimeError("Não foi possível localizar a raiz do projeto com src/data.")


def normalize_class_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized not in CLASS_ALIASES:
        raise ValueError(f"Classe desconhecida: {name!r}")
    return CLASS_ALIASES[normalized]


def natural_sort_key(value: str) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: str, length: int = 20) -> str:
    source = ":".join((DATASET_ID, *parts))
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}-{digest}"


def inspect_image(path: Path) -> dict[str, Any]:
    """Valida a imagem sem modificá-la e coleta metadados básicos."""
    try:
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
            image_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Imagem inválida ou corrompida: {path}") from error

    return {
        "width": width,
        "height": height,
        "mode": mode,
        "format": image_format,
        "file_size_bytes": path.stat().st_size,
    }


def discover_class_directories(split_directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}

    for directory in sorted(split_directory.iterdir()):
        if not directory.is_dir():
            continue
        try:
            label = normalize_class_name(directory.name)
        except ValueError:
            continue
        if label in result:
            raise ValueError(f"Duas pastas representam {label!r} em {split_directory}.")
        result[label] = directory

    missing = set(CLASS_TO_INDEX).difference(result)
    if missing:
        raise FileNotFoundError(
            f"Pastas de classe ausentes em {split_directory}: {sorted(missing)}"
        )
    return result


def duplicate_summaries(
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_hash[sample["content_sha256"]].append(sample)

    all_groups: list[dict[str, Any]] = []
    cross_split_groups: list[dict[str, Any]] = []

    for content_hash, group in sorted(by_hash.items()):
        if len(group) < 2:
            continue
        summary = {
            "content_sha256": content_hash,
            "count": len(group),
            "splits": sorted({item["original_split"] for item in group}),
            "samples": [
                {
                    "sample_id": item["sample_id"],
                    "original_split": item["original_split"],
                    "label": item["label"],
                    "relative_path": item["files"]["image"],
                }
                for item in group
            ],
        }
        all_groups.append(summary)
        if len(summary["splits"]) > 1:
            cross_split_groups.append(summary)

    return all_groups, cross_split_groups


def build_interim_manifest(dataset_root: Path, output_path: Path) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_root}")

    split_directories = {
        directory.name.lower(): directory
        for directory in dataset_root.iterdir()
        if directory.is_dir() and directory.name.lower() in EXPECTED_SPLITS
    }
    missing_splits = set(EXPECTED_SPLITS).difference(split_directories)
    if missing_splits:
        raise FileNotFoundError(f"Pastas de split ausentes: {sorted(missing_splits)}")

    samples: list[dict[str, Any]] = []
    corrupt_files: list[str] = []
    ignored_files: list[str] = []
    by_split = Counter()
    by_class = Counter()
    by_split_class: dict[str, Counter[str]] = defaultdict(Counter)
    dimensions = Counter()
    modes = Counter()
    formats = Counter()

    for split in EXPECTED_SPLITS:
        for label, class_directory in sorted(
            discover_class_directories(split_directories[split]).items()
        ):
            files = sorted(
                (path for path in class_directory.rglob("*") if path.is_file()),
                key=lambda path: natural_sort_key(
                    path.relative_to(dataset_root).as_posix()
                ),
            )

            for path in files:
                relative_path = path.relative_to(dataset_root).as_posix()
                if not is_image(path):
                    ignored_files.append(relative_path)
                    continue

                try:
                    metadata = inspect_image(path)
                except ValueError:
                    corrupt_files.append(relative_path)
                    continue

                sample_id = stable_id("big-sample", split, label, relative_path)
                case_id = stable_id("big-case", split, label, relative_path, length=16)
                content_hash = file_sha256(path)

                samples.append(
                    {
                        "sample_id": sample_id,
                        "case_id": case_id,
                        "patient_id": None,
                        "original_split": split,
                        "split": split,
                        "original_class_directory": class_directory.name,
                        "label_original": label,
                        "label": label,
                        "label_id": CLASS_TO_INDEX[label],
                        "eligible_for_binary_task": True,
                        "files": {"image": relative_path},
                        "content_sha256": content_hash,
                        "image_metadata": metadata,
                    }
                )

                by_split[split] += 1
                by_class[label] += 1
                by_split_class[split][label] += 1
                dimensions[f"{metadata['width']}x{metadata['height']}"] += 1
                modes[str(metadata["mode"])] += 1
                formats[str(metadata["format"])] += 1

    samples.sort(
        key=lambda item: (
            EXPECTED_SPLITS.index(item["original_split"]),
            item["label"],
            natural_sort_key(item["files"]["image"]),
        )
    )

    duplicate_groups, cross_split_duplicates = duplicate_summaries(samples)

    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "dataset_id": DATASET_ID,
        "dataset_stage": "interim",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "dataset_name": "Ultrasound Breast Images for Breast Cancer",
            "dataset_root_relative_path": "src/data/raw/big-dataset-9.248",
            "dataset_root_env": "BIG_DATASET_ROOT",
            "source_url": (
                "https://www.kaggle.com/datasets/vuppalaadithyasairam/"
                "ultrasound-breast-images-for-breast-cancer"
            ),
            "metadata_file": None,
            "patient_mapping_available": False,
            "source_splits_preserved": ["train", "val"],
        },
        "task_definition": {
            "final_task": "binary_image_classification",
            "target_classes": ["benign", "malignant"],
            "class_to_index": CLASS_TO_INDEX,
            "interim_labels_preserved": ["benign", "malignant"],
        },
        "statistics": {
            "total_valid_images": len(samples),
            "by_original_split": dict(sorted(by_split.items())),
            "by_class": dict(sorted(by_class.items())),
            "by_original_split_and_class": {
                split: dict(sorted(by_split_class[split].items()))
                for split in EXPECTED_SPLITS
            },
            "image_dimensions": dict(dimensions.most_common()),
            "image_modes": dict(modes.most_common()),
            "image_formats": dict(formats.most_common()),
            "exact_duplicate_group_count": len(duplicate_groups),
            "cross_split_duplicate_group_count": len(cross_split_duplicates),
            "corrupt_files": sorted(set(corrupt_files)),
            "ignored_non_image_files": sorted(set(ignored_files)),
        },
        "data_quality": {
            "exact_duplicate_groups": duplicate_groups,
            "cross_split_exact_duplicate_groups": cross_split_duplicates,
        },
        "limitations": [
            (
                "Não existe XLSX/CSV com patient_id ou metadados clínicos "
                "por imagem na estrutura utilizada."
            ),
            ("Sem patient_id, não é possível garantir separação por paciente."),
            "O interim preserva os splits train e val fornecidos pela fonte.",
        ],
        "samples": samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(f"Manifest interim criado em: {output_path}")
    print(json.dumps(manifest["statistics"], ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    project_root = find_project_root(Path(__file__))
    load_dotenv(project_root / ".env")

    configured_root = os.getenv("BIG_DATASET_ROOT")
    dataset_root = (
        Path(configured_root)
        if configured_root
        else project_root / "src" / "data" / "raw" / RAW_DIRECTORY_NAME
    )
    output_path = (
        project_root / "src" / "data" / "interim" / "manifests" / "big-dataset.json"
    )

    build_interim_manifest(dataset_root, output_path)
