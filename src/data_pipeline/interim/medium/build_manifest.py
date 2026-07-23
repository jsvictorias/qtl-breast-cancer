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

DATASET_ID = "medium-dataset"
RAW_DIRECTORY_NAME = "medium-dataset-780"

CLASS_TO_INDEX = {
    "benign": 0,
    "malignant": 1,
}

MASK_SUFFIX_PATTERN = re.compile(
    r"_mask(?:_\d+)?$",
    flags=re.IGNORECASE,
)

CASE_NUMBER_PATTERN = re.compile(r"\((\d+)\)")


def find_project_root(current_file: Path) -> Path:
    """Localiza a raiz do projeto procurando a estrutura src/data."""
    for parent in current_file.resolve().parents:
        if (parent / "src" / "data").is_dir():
            return parent

def natural_sort_key(value: str) -> list[Any]:
    """Ordena nomes com números de forma natural."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower()


def is_mask_file(path: Path) -> bool:
    return bool(MASK_SUFFIX_PATTERN.search(path.stem))

def create_case_id(label: str, image_path: Path) -> str:
    """Cria um identificador legível para o caso."""
    match = CASE_NUMBER_PATTERN.search(image_path.stem)

    if match:
        return f"{label}-{int(match.group(1)):04d}"

    digest = hashlib.sha256(image_path.stem.encode("utf-8")).hexdigest()[:10]

    return f"{label}-{digest}"


def create_sample_id(label: str, relative_path: str) -> str:
    """Cria um identificador determinístico para a amostra."""
    source = f"{DATASET_ID}:{label}:{relative_path}"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"medium-{digest}"


def inspect_image(path: Path) -> dict[str, Any]:
    """Valida a imagem e coleta metadados básicos sem alterá-la."""
    try:
        with Image.open(path) as image:
            width, height = image.size
            image_mode = image.mode
            image_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Imagem inválida ou corrompida: {path}") from error

    return {
        "width": width,
        "height": height,
        "mode": image_mode,
        "format": image_format,
        "file_size_bytes": path.stat().st_size,
    }


def discover_class_directories(dataset_root: Path) -> dict[str, Path]:
    """Localiza as três pastas de classe e normaliza seus nomes."""
    result: dict[str, Path] = {}

    for directory in sorted(dataset_root.iterdir()):
        if not directory.is_dir():
            continue

        try:
            normalized_label = normalize_class_name(directory.name)
        except ValueError:
            # Diretórios que não representam classes são ignorados.
            continue

        if normalized_label in result:
            raise ValueError(
                "Mais de uma pasta representa a classe "
                f"{normalized_label!r}: "
                f"{result[normalized_label]} e {directory}"
            )

        result[normalized_label] = directory

    missing = {"benign", "malignant", "normal"}.difference(result)

    if missing:
        raise FileNotFoundError(
            f"Pastas de classe ausentes no dataset médio: {sorted(missing)}"
        )

    return result


def build_interim_manifest(
    dataset_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """
    Cria o manifesto interim do BUSI/medium-dataset.

    As imagens originais viram amostras. Arquivos com sufixos
    _mask, _mask_1, _mask_2 etc. são associados à imagem
    correspondente e não são tratados como amostras independentes.
    """
    dataset_root = dataset_root.expanduser().resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset médio não encontrado: {dataset_root}")

    class_directories = discover_class_directories(dataset_root)

    samples: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    mask_counts_by_class: Counter[str] = Counter()

    missing_masks: list[str] = []
    multiple_masks: dict[str, list[str]] = {}
    corrupt_files: list[str] = []
    ignored_non_image_files: list[str] = []

    image_dimensions: Counter[str] = Counter()
    image_modes: Counter[str] = Counter()

    seen_relative_paths: set[str] = set()
    seen_sample_ids: set[str] = set()

    for label, class_directory in sorted(class_directories.items()):
        all_files = sorted(
            (path for path in class_directory.rglob("*") if path.is_file()),
            key=lambda path: natural_sort_key(
                path.relative_to(dataset_root).as_posix()
            ),
        )

        image_files = [path for path in all_files if is_image_file(path)]

        ignored_non_image_files.extend(
            path.relative_to(dataset_root).as_posix()
            for path in all_files
            if not is_image_file(path)
        )

        masks_by_base_stem: dict[str, list[Path]] = defaultdict(list)
        primary_images: list[Path] = []

        for image_path in image_files:
            if is_mask_file(image_path):
                masks_by_base_stem[mask_base_stem(image_path).lower()].append(
                    image_path
                )
            else:
                primary_images.append(image_path)

        for image_path in primary_images:
            relative_path = image_path.relative_to(dataset_root).as_posix()

            if relative_path in seen_relative_paths:
                raise ValueError(
                    f"Caminho de imagem duplicado no inventário: {relative_path}"
                )

            case_id = create_case_id(label=label, image_path=image_path)
            sample_id = create_sample_id(
                label=label,
                relative_path=relative_path,
            )

            if sample_id in seen_sample_ids:
                raise ValueError(f"sample_id duplicado: {sample_id}")

            seen_relative_paths.add(relative_path)
            seen_sample_ids.add(sample_id)

            try:
                image_metadata = inspect_image(image_path)
            except ValueError:
                corrupt_files.append(relative_path)
                continue

            image_dimensions[
                f"{image_metadata['width']}x{image_metadata['height']}"
            ] += 1
            image_modes[str(image_metadata["mode"])] += 1

            corresponding_masks = sorted(
                masks_by_base_stem.get(image_path.stem.lower(), []),
                key=lambda path: natural_sort_key(path.name),
            )

            valid_mask_paths: list[str] = []

            for mask_path in corresponding_masks:
                mask_relative_path = mask_path.relative_to(dataset_root).as_posix()

                try:
                    inspect_image(mask_path)
                except ValueError:
                    corrupt_files.append(mask_relative_path)
                    continue

                valid_mask_paths.append(mask_relative_path)

            if valid_mask_paths:
                mask_counts_by_class[label] += len(valid_mask_paths)

            # A classe normal normalmente não possui máscara de lesão.
            if label in {"benign", "malignant"} and not valid_mask_paths:
                missing_masks.append(case_id)

            if len(valid_mask_paths) > 1:
                multiple_masks[case_id] = valid_mask_paths

            eligible = label in CLASS_TO_INDEX

            samples.append(
                {
                    "sample_id": sample_id,
                    "case_id": case_id,
                    "patient_id": None,
                    "split": "unassigned",
                    "original_class_directory": class_directory.name,
                    "label_original": label,
                    "label": label,
                    "label_id": CLASS_TO_INDEX.get(label),
                    "eligible_for_binary_task": eligible,
                    "files": {
                        "image": relative_path,
                        "segmentation_masks": valid_mask_paths,
                    },
                    "image_metadata": image_metadata,
                }
            )

            class_counts[label] += 1

    samples.sort(
        key=lambda sample: (
            sample["label"],
            natural_sort_key(sample["case_id"]),
        )
    )

    expected_labels = {"benign", "malignant", "normal"}
    discovered_labels = set(class_counts)

    if discovered_labels != expected_labels:
        raise ValueError(
            "As classes inventariadas não correspondem às classes "
            "esperadas. "
            f"Encontradas: {sorted(discovered_labels)}"
        )

    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "dataset_id": DATASET_ID,
        "dataset_stage": "interim",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "dataset_name": "Breast Ultrasound Images Dataset (BUSI)",
            "dataset_root_relative_path": ("src/data/raw/medium-dataset-780"),
            "dataset_root_env": "MEDIUM_DATASET_ROOT",
            "source_url": (
                "https://www.kaggle.com/datasets/aryashah2k/"
                "breast-ultrasound-images-dataset"
            ),
            "reference_doi": "10.1016/j.dib.2019.104863",
            "metadata_file": None,
            "patient_mapping_available": False,
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
            "segmentation_masks_used_as_samples": False,
        },
        "statistics": {
            "total_primary_images": len(samples),
            "by_original_class": dict(sorted(class_counts.items())),
            "eligible_for_binary_task": sum(
                sample["eligible_for_binary_task"] for sample in samples
            ),
            "excluded_from_binary_task": sum(
                not sample["eligible_for_binary_task"] for sample in samples
            ),
            "total_valid_masks": sum(mask_counts_by_class.values()),
            "masks_by_class": dict(sorted(mask_counts_by_class.items())),
            "cases_without_expected_mask": missing_masks,
            "cases_with_multiple_masks": multiple_masks,
            "image_dimensions": dict(image_dimensions.most_common()),
            "image_modes": dict(image_modes.most_common()),
            "corrupt_files": sorted(set(corrupt_files)),
            "ignored_non_image_files": sorted(set(ignored_non_image_files)),
        },
        "limitations": [
            (
                "O diretório disponibilizado não contém um arquivo "
                "que relacione cada imagem a um patient_id confiável."
            ),
            (
                "O manifesto interim não cria splits; todos os casos "
                "permanecem como unassigned."
            ),
        ],
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

    return manifest


if __name__ == "__main__":
    project_root = find_project_root(Path(__file__))
    load_dotenv(project_root / ".env")

    root_from_env = os.getenv("MEDIUM_DATASET_ROOT")

    dataset_root = (
        Path(root_from_env)
        if root_from_env
        else (project_root / "src" / "data" / "raw" / RAW_DIRECTORY_NAME)
    )

    output_path = (
        project_root / "src" / "data" / "interim" / "manifests" / "medium-dataset.json"
    )

    build_interim_manifest(
        dataset_root=dataset_root,
        output_path=output_path,
    )
