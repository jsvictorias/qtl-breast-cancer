from collections import defaultdict
from pathlib import Path
from typing import Any

import imagehash
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_no_case_split_across_multiple_splits(samples: list[dict[str, Any]]) -> None:
    splits_by_case: dict[tuple[str, str], set[str]] = defaultdict(set)
    for sample in samples:
        splits_by_case[(sample["dataset"], sample["case_id"])].add(sample["split"])

    leaking = {key: splits for key, splits in splits_by_case.items() if len(splits) > 1}
    assert not leaking, (
        f"{len(leaking)} case(s) have samples spread across more than one split: "
        f"{leaking}"
    )


def test_every_sample_has_a_split(samples: list[dict[str, Any]]) -> None:
    missing = [s["sample_id"] for s in samples if not s["split"]]
    assert not missing, (
        f"{len(missing)} sample(s) were never assigned a split: {missing}"
    )


@pytest.mark.slow
def test_no_broken_image_or_mask_paths(samples: list[dict[str, Any]]) -> None:
    broken: list[str] = []
    for sample in samples:
        image_path = PROJECT_ROOT / sample["image_path"]
        if not image_path.is_file():
            broken.append(sample["image_path"])
        for mask_path in sample["mask_paths"]:
            if not (PROJECT_ROOT / mask_path).is_file():
                broken.append(mask_path)

    assert not broken, f"{len(broken)} broken path(s), e.g. {broken[:10]}"


def test_labels_within_vocabulary(processed_manifest: dict[str, Any]) -> None:
    vocab = processed_manifest["class_to_index"]
    samples = processed_manifest["samples"]

    bad_labels = [s["sample_id"] for s in samples if s["label"] not in vocab]
    assert not bad_labels, f"label outside vocabulary {vocab}: {bad_labels[:10]}"

    mismatched = [s["sample_id"] for s in samples if s["label_id"] != vocab[s["label"]]]
    assert not mismatched, f"label_id doesn't match class_to_index: {mismatched[:10]}"


@pytest.mark.slow
def test_no_perceptual_duplicate_images(samples: list[dict[str, Any]]) -> None:
    hash_to_sample_ids: dict[imagehash.ImageHash, list[str]] = defaultdict(list)
    missing: list[str] = []

    for sample in samples:
        image_path = PROJECT_ROOT / sample["image_path"]
        if not image_path.is_file():
            missing.append(sample["sample_id"])
            continue
        with Image.open(image_path) as image:
            phash = imagehash.phash(image)
        hash_to_sample_ids[phash].append(sample["sample_id"])

    if missing:
        pytest.skip(f"{len(missing)} image path(s) missing, e.g. {missing[:5]}")

    duplicate_groups = {
        str(phash): ids for phash, ids in hash_to_sample_ids.items() if len(ids) > 1
    }
    assert not duplicate_groups, (
        f"{len(duplicate_groups)} perceptual-hash duplicate group(s) found: "
        f"{dict(list(duplicate_groups.items())[:5])}"
    )
