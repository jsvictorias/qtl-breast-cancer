from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from classical.utils.label_map import LabelMap


@dataclass(frozen=True)
class ManifestBundle:
    """Conteúdo validado de um manifesto processed."""

    manifest_path: Path
    project_root: Path
    dataset_id: str
    dataset_root: Path
    manifest: dict[str, Any]
    samples: list[dict[str, Any]]
    label_map: LabelMap

    def samples_for_split(self, split: str) -> list[dict[str, Any]]:
        return [sample for sample in self.samples if sample["split"] == split]

    def split_counts(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for split in ("train", "val", "test"):
            labels = [item["label"] for item in self.samples_for_split(split)]
            result[split] = dict(sorted(Counter(labels).items()))
        return result


def _resolve_dataset_root(
    manifest: dict[str, Any],
    project_root: Path,
) -> Path:
    source = manifest.get("source", {})
    env_name = source.get("dataset_root_env")
    configured_root = os.getenv(str(env_name)) if env_name else None

    if configured_root:
        dataset_root = Path(configured_root).expanduser()
    else:
        relative_root = source.get("dataset_root_relative_path")
        if not relative_root:
            raise ValueError(
                "O manifesto não contém source.dataset_root_relative_path."
            )
        dataset_root = project_root / str(relative_root)

    dataset_root = dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            "Diretório raiz do dataset não encontrado: "
            f"{dataset_root}. Verifique o .env ou o campo "
            "source.dataset_root_relative_path do manifesto."
        )
    return dataset_root


def _extract_relative_path(sample: dict[str, Any]) -> str:
    direct_path = sample.get("source_relative_path")
    if direct_path:
        return str(direct_path)

    files = sample.get("files")
    if isinstance(files, dict) and files.get("image"):
        return str(files["image"])

    raise ValueError(
        "A amostra não possui source_relative_path nem files.image: "
        f"{sample.get('sample_id', '<sem sample_id>')}"
    )


def load_processed_manifest(
    manifest_path: str | Path,
    project_root: str | Path,
) -> ManifestBundle:
    """Carrega e valida o schema usado pelos três datasets processed."""

    project_root = Path(project_root).resolve()
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    manifest_path = manifest_path.resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifesto não encontrado: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest.get("dataset_stage") != "processed":
        raise ValueError(
            f"O manifesto {manifest_path} não está no estágio 'processed'."
        )

    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("O manifesto precisa conter uma lista não vazia em samples.")

    task_definition = manifest.get("task_definition", {})
    class_to_index = task_definition.get("class_to_index")
    if not isinstance(class_to_index, dict) or len(class_to_index) < 2:
        labels = sorted({str(item.get("label")) for item in raw_samples})
        class_to_index = {label: index for index, label in enumerate(labels)}

    label_map = LabelMap.from_mapping(class_to_index)

    validated_samples: list[dict[str, Any]] = []
    valid_splits = {"train", "val", "test"}

    for index, sample in enumerate(raw_samples):
        if not isinstance(sample, dict):
            raise TypeError(f"samples[{index}] não é um objeto JSON.")

        split = str(sample.get("split", "")).strip().lower()
        label = str(sample.get("label", "")).strip()

        if split not in valid_splits:
            raise ValueError(
                f"Split inválido em samples[{index}]: {split!r}. "
                "Esperado train, val ou test."
            )
        if label not in label_map.class_to_index:
            raise ValueError(
                f"Classe {label!r} não existe em task_definition.class_to_index."
            )

        relative_path = _extract_relative_path(sample)
        normalized = dict(sample)
        normalized["split"] = split
        normalized["label"] = label
        normalized["source_relative_path"] = relative_path
        validated_samples.append(normalized)

    dataset_root = _resolve_dataset_root(manifest, project_root)

    bundle = ManifestBundle(
        manifest_path=manifest_path,
        project_root=project_root,
        dataset_id=str(manifest.get("dataset_id", manifest_path.parent.name)),
        dataset_root=dataset_root,
        manifest=manifest,
        samples=validated_samples,
        label_map=label_map,
    )

    for split in ("train", "val", "test"):
        if not bundle.samples_for_split(split):
            raise ValueError(f"O manifesto não possui amostras no split {split!r}.")

    train_labels = {item["label"] for item in bundle.samples_for_split("train")}
    missing_in_train = set(label_map.class_to_index).difference(train_labels)
    if missing_in_train:
        raise ValueError(
            f"Existem classes sem amostras no treino: {sorted(missing_in_train)}."
        )

    return bundle


class ManifestImageDataset(Dataset[dict[str, Any]]):
    """Dataset PyTorch que abre as imagens apontadas pelo manifest.json."""

    def __init__(
        self,
        bundle: ManifestBundle,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.bundle = bundle
        self.split = split
        self.transform = transform
        self.samples = bundle.samples_for_split(split)

        if not self.samples:
            raise ValueError(f"Nenhuma amostra encontrada para o split {split!r}.")

    def __len__(self) -> int:
        return len(self.samples)

    def _image_path(self, sample: dict[str, Any]) -> Path:
        relative_path = Path(str(sample["source_relative_path"]))
        image_path = (
            relative_path
            if relative_path.is_absolute()
            else self.bundle.dataset_root / relative_path
        ).resolve()

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Imagem não encontrada: {image_path} "
                f"(sample_id={sample.get('sample_id')})"
            )
        return image_path

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_path = self._image_path(sample)

        try:
            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise RuntimeError(f"Falha ao abrir a imagem: {image_path}") from error

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            raise RuntimeError("Um transform que gere tensor é obrigatório.")

        label = str(sample["label"])
        target = self.bundle.label_map.encode(label)

        return {
            "image": image_tensor,
            "target": target,
            "label": label,
            "path": str(image_path),
            "sample_id": str(sample.get("sample_id", index)),
        }

    def targets(self) -> list[int]:
        return [
            self.bundle.label_map.encode(str(sample["label"]))
            for sample in self.samples
        ]
