from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

DATASET_ID = "big-dataset"
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
REMOVE_EXACT_DUPLICATES = True
ALLOWED_LABELS = {"benign", "malignant"}
CLASS_TO_INDEX = {"benign": 0, "malignant": 1}


def find_project_root(current_file: Path) -> Path:
    for parent in current_file.resolve().parents:
        if (parent / "src" / "data").is_dir():
            return parent
    raise RuntimeError("Não foi possível localizar a raiz do projeto com src/data.")


def validate_interim(manifest: dict[str, Any]) -> None:
    if manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("O manifesto não pertence ao big-dataset.")
    if manifest.get("dataset_stage") != "interim":
        raise ValueError("O manifesto de entrada não está no estágio interim.")
    if not isinstance(manifest.get("samples"), list):
        raise ValueError("O campo samples está ausente ou não é uma lista.")


def validate_ratios() -> None:
    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"As proporções devem somar 1.0, mas somaram {total}.")


def count_by_split_and_label(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        selected = [item for item in samples if item["split"] == split]
        result[split] = {
            "total": len(selected),
            "by_label": dict(
                sorted(Counter(item["label"] for item in selected).items())
            ),
        }
    return result


def deduplicate(
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove cópias exatas, preferindo train original e depois val."""
    if not REMOVE_EXACT_DUPLICATES:
        return samples, []

    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_hash[sample["content_sha256"]].append(sample)

    priority = {"train": 0, "val": 1}
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for content_hash, group in by_hash.items():
        ordered = sorted(
            group,
            key=lambda item: (
                priority[item["original_split"]],
                item["files"]["image"],
            ),
        )
        retained = ordered[0]
        kept.append(retained)

        for duplicate in ordered[1:]:
            excluded.append(
                {
                    "sample_id": duplicate["sample_id"],
                    "case_id": duplicate["case_id"],
                    "label": duplicate["label"],
                    "original_split": duplicate["original_split"],
                    "source_relative_path": duplicate["files"]["image"],
                    "content_sha256": content_hash,
                    "reason": "exact_duplicate_content",
                    "retained_sample_id": retained["sample_id"],
                    "retained_relative_path": retained["files"]["image"],
                }
            )

    return kept, excluded


def create_standard_splits(samples: list[dict[str, Any]]) -> dict[str, str]:
    """Recria splits 70/15/15 estratificados para padronizar os datasets."""
    labels = [item["label"] for item in samples]
    counts = Counter(labels)
    if set(counts) != ALLOWED_LABELS:
        raise ValueError(f"As duas classes são obrigatórias: {dict(counts)}")

    indices = list(range(len(samples)))
    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    temporary_labels = [labels[index] for index in temporary_indices]
    relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=relative_test_ratio,
        random_state=RANDOM_SEED,
        stratify=temporary_labels,
    )

    result: dict[str, str] = {}
    for index in train_indices:
        result[samples[index]["sample_id"]] = "train"
    for index in val_indices:
        result[samples[index]["sample_id"]] = "val"
    for index in test_indices:
        result[samples[index]["sample_id"]] = "test"
    return result


def validate_final(samples: list[dict[str, Any]]) -> None:
    sample_ids = [item["sample_id"] for item in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Existem sample_ids duplicados no processed.")

    hashes_by_split: dict[str, set[str]] = defaultdict(set)
    for item in samples:
        if item["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Split inválido: {item['split']}")
        if item["label"] not in ALLOWED_LABELS:
            raise ValueError(f"Classe inválida: {item['label']}")
        hashes_by_split[item["split"]].add(item["content_sha256"])

    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if hashes_by_split[first] & hashes_by_split[second]:
            raise ValueError(f"Existem duplicatas exatas entre {first} e {second}.")


def build_processed_manifest(
    interim_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    validate_ratios()
    if not interim_manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifesto interim não encontrado: {interim_manifest_path}"
        )

    with interim_manifest_path.open("r", encoding="utf-8") as file:
        interim = json.load(file)
    validate_interim(interim)

    source_samples = [
        item
        for item in interim["samples"]
        if item.get("eligible_for_binary_task") is True
        and item.get("label") in ALLOWED_LABELS
    ]
    if not source_samples:
        raise ValueError("Nenhuma amostra elegível foi encontrada.")

    deduplicated, duplicate_exclusions = deduplicate(source_samples)
    split_map = create_standard_splits(deduplicated)

    processed_samples: list[dict[str, Any]] = []
    for item in deduplicated:
        processed_samples.append(
            {
                "sample_id": item["sample_id"],
                "case_id": item["case_id"],
                "patient_id": item.get("patient_id"),
                "original_split": item["original_split"],
                "split": split_map[item["sample_id"]],
                "label": item["label"],
                "label_id": item["label_id"],
                "source_relative_path": item["files"]["image"],
                "content_sha256": item["content_sha256"],
                "image_metadata": item.get("image_metadata", {}),
            }
        )

    split_order = {"train": 0, "val": 1, "test": 2}
    processed_samples.sort(
        key=lambda item: (
            split_order[item["split"]],
            item["label"],
            item["source_relative_path"],
        )
    )
    validate_final(processed_samples)

    overall = Counter(item["label"] for item in processed_samples)
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "dataset_id": DATASET_ID,
        "dataset_stage": "processed",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "interim_manifest": "src/data/interim/manifests/big-dataset.json",
            "dataset_root_env": "BIG_DATASET_ROOT",
            "dataset_root_relative_path": "src/data/raw/big-dataset-9.248",
        },
        "task_definition": {
            "task": "binary_image_classification",
            "target_classes": ["benign", "malignant"],
            "class_to_index": CLASS_TO_INDEX,
            "model_input": "original_ultrasound_image",
        },
        "split_config": {
            "strategy": "stratified_resplit_all_source_images",
            "original_split_preserved_as_metadata": True,
            "random_seed": RANDOM_SEED,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "stratified_by_label": True,
            "grouping_unit": "image_sample",
            "patient_level_split": False,
            "limitation": (
                "Não há patient_id por arquivo; portanto, a separação não "
                "pode ser garantida no nível da paciente."
            ),
        },
        "deduplication": {
            "enabled": REMOVE_EXACT_DUPLICATES,
            "method": "sha256_exact_content",
            "retention_priority": ["original_train", "original_val"],
            "excluded_duplicate_count": len(duplicate_exclusions),
        },
        "lenet_preprocessing": {
            "image_mode": "grayscale",
            "channels": 1,
            "resize": [32, 32],
            "tensor_scaling": "0_to_1",
            "normalization": {
                "mean": [0.5],
                "std": [0.5],
                "output_range_approximately": [-1.0, 1.0],
            },
            "train_augmentation": [
                {"name": "RandomHorizontalFlip", "probability": 0.5},
                {"name": "RandomRotation", "degrees": 10},
            ],
            "validation_and_test_augmentation": [],
        },
        "statistics": {
            "total_in_interim": len(interim["samples"]),
            "total_before_deduplication": len(source_samples),
            "total_in_processed": len(processed_samples),
            "excluded_exact_duplicates": len(duplicate_exclusions),
            "overall_by_label": dict(sorted(overall.items())),
            "by_split": count_by_split_and_label(processed_samples),
        },
        "excluded_samples": duplicate_exclusions,
        "samples": processed_samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(f"Manifest processed criado em: {output_path}")
    print(json.dumps(manifest["statistics"], ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    project_root = find_project_root(Path(__file__))
    interim_path = (
        project_root / "src" / "data" / "interim" / "manifests" / "big-dataset.json"
    )
    output_path = (
        project_root / "src" / "data" / "processed" / "big-dataset" / "manifest.json"
    )
    build_processed_manifest(interim_path, output_path)
