from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

DATASET_ID = "medium-dataset"

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

ALLOWED_LABELS = {
    "benign",
    "malignant",
}


def find_project_root(current_file: Path) -> Path:
    """Localiza a raiz do projeto procurando src/data."""
    for parent in current_file.resolve().parents:
        if (parent / "src" / "data").is_dir():
            return parent

    raise RuntimeError(
        "Não foi possível localizar a raiz do projeto. "
        "A estrutura esperada contém src/data."
    )


def count_by_split_and_label(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calcula a distribuição de classes em cada split."""
    result: dict[str, Any] = {}

    for split in ("train", "val", "test"):
        split_samples = [sample for sample in samples if sample["split"] == split]

        result[split] = {
            "total": len(split_samples),
            "by_label": dict(
                sorted(Counter(sample["label"] for sample in split_samples).items())
            ),
        }

    return result


def validate_ratios() -> None:
    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO

    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            "TRAIN_RATIO + VAL_RATIO + TEST_RATIO "
            f"deve ser 1.0, mas resultou em {total}."
        )


def validate_interim_manifest(
    manifest: dict[str, Any],
) -> None:
    """Confere se o arquivo de entrada é o interim esperado."""
    if manifest.get("dataset_id") != DATASET_ID:
        raise ValueError(
            f"O manifesto informado não pertence ao dataset {DATASET_ID!r}."
        )

    if manifest.get("dataset_stage") != "interim":
        raise ValueError("O manifesto informado não está no estágio interim.")

    if not isinstance(manifest.get("samples"), list):
        raise ValueError("O campo samples está ausente ou não é uma lista.")


def split_samples(
    eligible_samples: list[dict[str, Any]],
) -> dict[str, str]:
    """
    Cria splits estratificados por classe.

    Como o dataset baixado não fornece patient_id por imagem,
    a divisão é realizada no nível da imagem.
    """
    indices = list(range(len(eligible_samples)))
    labels = [eligible_samples[index]["label"] for index in indices]

    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    temporary_labels = [eligible_samples[index]["label"] for index in temporary_indices]

    relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

    val_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=relative_test_ratio,
        random_state=RANDOM_SEED,
        stratify=temporary_labels,
    )

    split_by_sample_id: dict[str, str] = {}

    for index in train_indices:
        split_by_sample_id[eligible_samples[index]["sample_id"]] = "train"

    for index in val_indices:
        split_by_sample_id[eligible_samples[index]["sample_id"]] = "val"

    for index in test_indices:
        split_by_sample_id[eligible_samples[index]["sample_id"]] = "test"

    return split_by_sample_id


def validate_final_splits(
    processed_samples: list[dict[str, Any]],
) -> None:
    """Garante que cada amostra aparece em apenas um split."""
    sample_ids = [sample["sample_id"] for sample in processed_samples]

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Existem sample_ids duplicados no processed.")

    invalid_splits = {
        sample["split"]
        for sample in processed_samples
        if sample["split"]
        not in {
            "train",
            "val",
            "test",
        }
    }

    if invalid_splits:
        raise ValueError(f"Splits inválidos encontrados: {invalid_splits}")

    invalid_labels = {
        sample["label"]
        for sample in processed_samples
        if sample["label"] not in ALLOWED_LABELS
    }

    if invalid_labels:
        raise ValueError(
            f"Classes não permitidas no processed: {sorted(invalid_labels)}"
        )


def build_processed_manifest(
    interim_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """
    Converte o manifesto interim em um manifesto pronto
    para o baseline LeNet e para experimentos posteriores.
    """
    validate_ratios()

    if not interim_manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifesto interim não encontrado: {interim_manifest_path}"
        )

    with interim_manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        interim = json.load(file)

    validate_interim_manifest(interim)

    eligible_samples = [
        sample
        for sample in interim["samples"]
        if (
            sample.get("eligible_for_binary_task") is True
            and sample.get("label") in ALLOWED_LABELS
        )
    ]

    excluded_samples = [
        {
            "sample_id": sample["sample_id"],
            "case_id": sample["case_id"],
            "label_original": sample.get("label_original"),
            "source_relative_path": sample["files"]["image"],
            "reason": ("normal_class_excluded_from_binary_classification"),
        }
        for sample in interim["samples"]
        if sample.get("label") == "normal"
    ]

    if not eligible_samples:
        raise ValueError("Nenhuma amostra benign/malignant foi encontrada.")

    class_counts = Counter(sample["label"] for sample in eligible_samples)

    if set(class_counts) != ALLOWED_LABELS:
        raise ValueError(
            f"O processed requer as duas classes. Encontradas: {dict(class_counts)}"
        )

    split_by_sample_id = split_samples(eligible_samples)

    processed_samples: list[dict[str, Any]] = []

    for sample in eligible_samples:
        processed_samples.append(
            {
                "sample_id": sample["sample_id"],
                "case_id": sample["case_id"],
                "patient_id": sample.get("patient_id"),
                "split": split_by_sample_id[sample["sample_id"]],
                "label": sample["label"],
                "label_id": sample["label_id"],
                "source_relative_path": (sample["files"]["image"]),
                "segmentation_mask_relative_paths": (
                    sample["files"].get(
                        "segmentation_masks",
                        [],
                    )
                ),
                "image_metadata": sample.get(
                    "image_metadata",
                    {},
                ),
            }
        )

    split_order = {
        "train": 0,
        "val": 1,
        "test": 2,
    }

    processed_samples.sort(
        key=lambda sample: (
            split_order[sample["split"]],
            sample["label"],
            sample["case_id"],
        )
    )

    validate_final_splits(processed_samples)

    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "dataset_id": DATASET_ID,
        "dataset_stage": "processed",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "interim_manifest": ("src/data/interim/manifests/medium-dataset.json"),
            "dataset_root_env": "MEDIUM_DATASET_ROOT",
            "dataset_root_relative_path": ("src/data/raw/medium-dataset-780"),
        },
        "task_definition": {
            "task": "binary_image_classification",
            "target_classes": ["benign", "malignant"],
            "class_to_index": {
                "benign": 0,
                "malignant": 1,
            },
            "model_input": "original_ultrasound_image",
            "segmentation_masks_used_as_model_input": False,
        },
        "split_config": {
            "strategy": "stratified_holdout_by_image",
            "random_seed": RANDOM_SEED,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "grouping_unit": "image_sample",
            "patient_level_split": False,
            "limitation": (
                "O arquivo baixado não fornece um patient_id "
                "confiável para cada imagem. Portanto, não é "
                "possível garantir que imagens da mesma paciente "
                "estejam em apenas um split."
            ),
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
                {
                    "name": "RandomHorizontalFlip",
                    "probability": 0.5,
                },
                {
                    "name": "RandomRotation",
                    "degrees": 10,
                },
            ],
            "validation_and_test_augmentation": [],
        },
        "statistics": {
            "total_in_interim": len(interim["samples"]),
            "total_in_processed": len(processed_samples),
            "excluded_total": len(excluded_samples),
            "overall_by_label": dict(sorted(class_counts.items())),
            "by_split": count_by_split_and_label(processed_samples),
        },
        "excluded_samples": excluded_samples,
        "samples": processed_samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Manifest processed criado em: {output_path}")
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

    interim_manifest_path = (
        project_root / "src" / "data" / "interim" / "manifests" / "medium-dataset.json"
    )

    output_path = (
        project_root / "src" / "data" / "processed" / "medium-dataset" / "manifest.json"
    )

    build_processed_manifest(
        interim_manifest_path=interim_manifest_path,
        output_path=output_path,
    )
