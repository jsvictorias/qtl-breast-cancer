from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def count_by_split_and_label(samples: list[dict]) -> dict:
    result = {}

    for split in ("train", "val", "test"):
        split_samples = [sample for sample in samples if sample["split"] == split]

        result[split] = {
            "total": len(split_samples),
            "by_label": dict(
                sorted(Counter(sample["label"] for sample in split_samples).items())
            ),
        }

    return result


def build_processed_manifest(
    interim_manifest_path: Path,
    output_path: Path,
) -> None:
    with interim_manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        interim = json.load(file)

    eligible = [
        sample
        for sample in interim["samples"]
        if sample.get("eligible_for_binary_task") is True
        and sample.get("label") in {"benign", "malignant"}
    ]

    excluded = [
        {
            "sample_id": sample["sample_id"],
            "case_id": sample["case_id"],
            "label_original": sample["label_original"],
            "reason": ("normal_class_excluded_from_binary_classification"),
        }
        for sample in interim["samples"]
        if sample.get("label") == "normal"
    ]

    indices = list(range(len(eligible)))
    labels = [eligible[index]["label"] for index in indices]

    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    temporary_labels = [eligible[index]["label"] for index in temporary_indices]

    relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

    val_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=relative_test_ratio,
        random_state=RANDOM_SEED,
        stratify=temporary_labels,
    )

    split_by_index = {
        **{index: "train" for index in train_indices},
        **{index: "val" for index in val_indices},
        **{index: "test" for index in test_indices},
    }

    processed_samples = []

    for index, sample in enumerate(eligible):
        processed_samples.append(
            {
                "sample_id": sample["sample_id"],
                "case_id": sample["case_id"],
                "split": split_by_index[index],
                "label": sample["label"],
                "label_id": sample["label_id"],
                "source_relative_path": (sample["files"]["image"]),
                "tumor_mask_relative_path": (sample["files"]["tumor_mask"]),
                "other_mask_relative_path": (sample["files"]["other_mask"]),
                "clinical_metadata": (sample["clinical_metadata"]),
            }
        )

    processed_samples.sort(
        key=lambda sample: (
            {
                "train": 0,
                "val": 1,
                "test": 2,
            }[sample["split"]],
            sample["case_id"],
        )
    )

    manifest = {
        "schema_version": "1.0.0",
        "dataset_id": "small-dataset",
        "dataset_stage": "processed",
        "generated_at_utc": (datetime.now(UTC).isoformat()),
        "source": {
            "interim_manifest": ("src/data/interim/manifests/small-dataset.json"),
            "dataset_root_env": "SMALL_DATASET_ROOT",
            "dataset_root_relative_path": ("src/data/raw/small-dataset"),
        },
        "task_definition": {
            "task": "binary_image_classification",
            "target_classes": [
                "benign",
                "malignant",
            ],
            "class_to_index": {
                "benign": 0,
                "malignant": 1,
            },
            "model_input": ("original_ultrasound_image"),
            "tumor_mask_used_as_model_input": False,
        },
        "split_config": {
            "strategy": "stratified_holdout",
            "random_seed": RANDOM_SEED,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": TEST_RATIO,
            "grouping_unit": "case_id",
        },
        "lenet_preprocessing": {
            "image_mode": "grayscale",
            "channels": 1,
            "resize": [32, 32],
            "tensor_scaling": "0_to_1",
            "normalization": {
                "mean": [0.5],
                "std": [0.5],
                "output_range_approximately": [
                    -1.0,
                    1.0,
                ],
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
            "excluded_total": len(excluded),
            "overall_by_label": dict(
                sorted(Counter(sample["label"] for sample in processed_samples).items())
            ),
            "by_split": count_by_split_and_label(processed_samples),
        },
        "excluded_samples": excluded,
        "samples": processed_samples,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Manifest criado em: {output_path}")
    print(
        json.dumps(
            manifest["statistics"],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    interim_path = (
        project_root / "src" / "data" / "interim" / "manifests" / "small-dataset.json"
    )

    output_path = (
        project_root / "src" / "data" / "processed" / "small-dataset" / "manifest.json"
    )

    build_processed_manifest(
        interim_manifest_path=interim_path,
        output_path=output_path,
    )
