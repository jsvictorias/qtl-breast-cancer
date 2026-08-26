import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLASS_TO_INDEX = {"benign": 0, "malignant": 1}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_SEED = 42


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_samples(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_id = manifest["dataset_id"]
    return [
        {
            "sample_id": sample["sample_id"],
            "dataset": dataset_id,
            "case_id": sample["case_id"],
            "label": sample["label"],
            "label_id": CLASS_TO_INDEX[sample["label"]],
            "split": None,
            "image_path": sample["image_relative_path"],
            "mask_paths": sample["mask_relative_paths"],
            "metadata": sample["metadata"],
            "group_key": sample["group_key"],
        }
        for sample in manifest["samples"]
        if sample["label"] != "normal"
    ]


def dedupe_by_image_content(
    samples: list[dict[str, Any]], project_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        content_hash = hashlib.md5(
            (project_root / sample["image_path"]).read_bytes()
        ).hexdigest()
        groups[content_hash].append(sample)

    kept: list[dict[str, Any]] = []
    conflicting_label_dropped: list[str] = []
    duplicate_dropped: list[str] = []
    conflicting_groups = 0
    collapsed_groups = 0

    for group in groups.values():
        if len(group) == 1:
            kept.append(group[0])
            continue

        if len({sample["label"] for sample in group}) > 1:
            conflicting_groups += 1
            conflicting_label_dropped.extend(sample["sample_id"] for sample in group)
            continue

        collapsed_groups += 1
        group_sorted = sorted(group, key=lambda sample: sample["sample_id"])
        kept.append(group_sorted[0])
        duplicate_dropped.extend(sample["sample_id"] for sample in group_sorted[1:])

    report = {
        "conflicting_label_groups_dropped": conflicting_groups,
        "conflicting_label_samples_dropped": sorted(conflicting_label_dropped),
        "same_label_duplicate_groups_collapsed": collapsed_groups,
        "same_label_duplicates_dropped": sorted(duplicate_dropped),
    }
    return kept, report


def stratified_group_split(
    samples: list[dict[str, Any]], ratios: dict[str, float], seed: int
) -> None:
    rng = random.Random(seed)
    strata: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample in samples:
        strata[(sample["dataset"], sample["label"])][sample["group_key"]].append(sample)

    for groups_by_key in strata.values():
        groups = list(groups_by_key.items())
        rng.shuffle(groups)
        groups.sort(key=lambda group: len(group[1]), reverse=True)

        total = sum(len(group_samples) for _, group_samples in groups)
        targets = {split: ratio * total for split, ratio in ratios.items()}
        current = dict.fromkeys(ratios, 0.0)

        for _, group_samples in groups:
            deficit = {split: targets[split] - current[split] for split in ratios}
            chosen_split = max(deficit, key=deficit.__getitem__)
            for sample in group_samples:
                sample["split"] = chosen_split
            current[chosen_split] += len(group_samples)


def build_processed_manifest(
    small_path: Path, medium_path: Path, big_path: Path, output_path: Path
) -> None:
    samples: list[dict[str, Any]] = []
    for path in (small_path, medium_path, big_path):
        samples.extend(_normalize_samples(_load_manifest(path)))

    samples, dedup_report = dedupe_by_image_content(samples, PROJECT_ROOT)

    stratified_group_split(samples, SPLIT_RATIOS, SPLIT_SEED)

    for sample in samples:
        del sample["group_key"]

    dataset_ids = sorted({sample["dataset"] for sample in samples})

    manifest = {
        "dataset_stage": "processed",
        "class_to_index": CLASS_TO_INDEX,
        "split_strategy": (
            f"stratified group split per (dataset, label), seed={SPLIT_SEED}, "
            f"ratios={SPLIT_RATIOS}."
        ),
        "content_dedup": dedup_report,
        "total_samples": len(samples),
        "dataset_counts": dict(Counter(sample["dataset"] for sample in samples)),
        "label_counts": dict(Counter(sample["label"] for sample in samples)),
        "split_counts": dict(Counter(sample["split"] for sample in samples)),
        "dataset_split_counts": {
            dataset_id: dict(
                Counter(
                    sample["split"]
                    for sample in samples
                    if sample["dataset"] == dataset_id
                )
            )
            for dataset_id in dataset_ids
        },
        "samples": samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def main() -> None:
    interim_dir = PROJECT_ROOT / "data" / "interim"
    output_path = PROJECT_ROOT / "data" / "processed" / "manifest.json"

    build_processed_manifest(
        interim_dir / "manifest_small.json",
        interim_dir / "manifest_medium.json",
        interim_dir / "manifest_big.json",
        output_path,
    )


if __name__ == "__main__":
    main()
