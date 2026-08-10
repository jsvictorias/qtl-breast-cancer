# ruff: noqa
from __future__ import annotations
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

CLASS_TO_INDEX = {
    "benign": 0,
    "malignant": 1,
}

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}
SPLIT_SEED = 42

B_FILENAME_PATTERN = re.compile(
    r"^(?P<cls>benign|malignant)\s*\((?P<base_id>\d+)\)(?:\s*-\s*(?P<aug_chain>.+))?\.png$"
)


def load_interim_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_small_samples(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_id = manifest["dataset_id"]
    normalized: list[dict[str, Any]] = []

    for sample in manifest["samples"]:
        label = sample["label"]
        normalized.append(
            {
                "sample_id": sample["sample_id"],
                "dataset_id": dataset_id,
                "label": label,
                "label_id": CLASS_TO_INDEX.get(label),
                "split": None,
                "source_relative_path": (
                    f"src/data/raw/small-dataset/{sample['image_filename']}"
                ),
                "group_key": f"small-{sample['case_id']}",
                "augmentations": [],
                "is_original": True,
            }
        )
    return normalized


def normalize_medium_samples(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_id = manifest["dataset_id"]
    normalized: list[dict[str, Any]] = []

    for sample in manifest["samples"]:
        label = sample["label"]
        normalized.append(
            {
                "sample_id": sample["sample_id"],
                "dataset_id": dataset_id,
                "label": label,
                "label_id": sample["label_id"],
                "split": None,
                "source_relative_path": (
                    f"src/data/raw/medium-dataset-780/{label}/{sample['image_filename']}"
                ),
                "group_key": f"medium-{label}-{sample['image_filename']}",
                "augmentations": [],
                "is_original": True,
            }
        )
    return normalized


def normalize_big_samples(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_id = manifest["dataset_id"]
    normalized: list[dict[str, Any]] = []

    for sample in manifest["samples"]:
        label = sample["label"]
        split = sample["split"]

        match = B_FILENAME_PATTERN.match(sample["filename"])
        base_id = match.group("base_id") if match else sample["filename"]

        normalized.append(
            {
                "sample_id": sample["sample_id"],
                "dataset_id": dataset_id,
                "label": label,
                "label_id": CLASS_TO_INDEX.get(label),
                "split": None,
                "source_relative_path": (
                    f"src/data/raw/big-dataset-9.248/{split}/{label}/{sample['filename']}"
                ),
                "group_key": f"big-{label}-{base_id}",
                "augmentations": sample["augmentations"],
                "is_original": sample["is_original"],
            }
        )
    return normalized


def dedupe_redundant_augmentations(
    samples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    kept: list[dict[str, Any]] = []
    discarded = 0

    for sample in samples:
        combo_key = (sample["group_key"], tuple(sorted(sample["augmentations"])))
        if combo_key in seen:
            discarded += 1
            continue
        seen.add(combo_key)
        kept.append(sample)

    return kept, discarded


def stratified_group_split(
    samples: list[dict[str, Any]], ratios: dict[str, float], seed: int
) -> None:
    rng = random.Random(seed)

    by_label_groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample in samples:
        by_label_groups[sample["label"]][sample["group_key"]].append(sample)

    for label, groups_dict in by_label_groups.items():
        groups = list(groups_dict.items())  # [(group_key, [samples...]), ...]
        rng.shuffle(groups)
        groups.sort(key=lambda g: len(g[1]), reverse=True)

        total = sum(len(g[1]) for g in groups)
        targets = {split: ratio * total for split, ratio in ratios.items()}
        current = {split: 0 for split in ratios}

        for _, group_samples in groups:
            deficit = {split: targets[split] - current[split] for split in ratios}
            chosen_split = max(deficit, key=deficit.__getitem__)
            for sample in group_samples:
                sample["split"] = chosen_split
            current[chosen_split] += len(group_samples)


def build_processed_manifest(
    s_interim_path: Path,
    m_interim_path: Path,
    b_interim_path: Path,
    output_path: Path,
) -> None:
    s_manifest = load_interim_manifest(s_interim_path)
    m_manifest = load_interim_manifest(m_interim_path)
    b_manifest = load_interim_manifest(b_interim_path)

    s_samples = normalize_small_samples(s_manifest)
    m_samples = normalize_medium_samples(m_manifest)
    b_samples = normalize_big_samples(b_manifest)

    b_samples, b_discarded_dupes = dedupe_redundant_augmentations(b_samples)

    all_samples = s_samples + m_samples + b_samples

    stratified_group_split(all_samples, SPLIT_RATIOS, SPLIT_SEED)

    for sample in all_samples:
        del sample["group_key"]

    split_counts: Counter[str] = Counter(sample["split"] for sample in all_samples)
    label_counts: Counter[str] = Counter(sample["label"] for sample in all_samples)
    dataset_counts: Counter[str] = Counter(
        sample["dataset_id"] for sample in all_samples
    )
    original_vs_augmented = Counter(
        "original" if sample["is_original"] else "augmented" for sample in all_samples
    )

    manifest = {
        "dataset_stage": "processed",
        "split_strategy": (
            f"stratified group split by label, seed={SPLIT_SEED}, ratios={SPLIT_RATIOS}. "
            "Grupo = imagem-base (original + suas augmentations); grupo inteiro "
            "vai pro mesmo split, evitando leakage entre variações da mesma imagem."
        ),
        "augmentations_policy": (
            "augmentations mantidas como amostras válidas (análise de escala de dados). "
            "Combinações de augmentation duplicadas/redundantes por imagem-base "
            "foram removidas (dedupe)."
        ),
        "big_dataset_augmentation_dupes_discarded": b_discarded_dupes,
        "total_samples": len(all_samples),
        "split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "dataset_counts": dict(dataset_counts),
        "original_vs_augmented_counts": dict(original_vs_augmented),
        "samples": all_samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    load_dotenv()
    project_root = Path(__file__).resolve().parents[2]

    s_interim_path = project_root / "src" / "data" / "interim" / "manifest_small.json"
    m_interim_path = project_root / "src" / "data" / "interim" / "manifest_medium.json"
    b_interim_path = project_root / "src" / "data" / "interim" / "manifest_big.json"

    output_path = (
        project_root / "src" / "data" / "processed" / "manifest_processed.json"
    )

    build_processed_manifest(
        s_interim_path,
        m_interim_path,
        b_interim_path,
        output_path,
    )
