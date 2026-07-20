from __future__ import annotations

from typing import Any

from torchvision import transforms


def _normalization(preprocessing: dict[str, Any]) -> tuple[list[float], list[float]]:
    normalization = preprocessing.get("normalization", {})
    mean = [float(value) for value in normalization.get("mean", [0.5])]
    std = [float(value) for value in normalization.get("std", [0.5])]

    if len(mean) != len(std):
        raise ValueError("mean e std precisam ter o mesmo número de canais.")
    if any(value <= 0 for value in std):
        raise ValueError("Todos os valores de std precisam ser positivos.")
    return mean, std


def _augmentation_from_manifest(item: dict[str, Any]):
    name = str(item.get("name", "")).strip()

    if name == "RandomHorizontalFlip":
        return transforms.RandomHorizontalFlip(p=float(item.get("probability", 0.5)))
    if name == "RandomRotation":
        return transforms.RandomRotation(degrees=float(item.get("degrees", 10)))

    raise ValueError(
        f"Augmentation não suportada no manifesto: {name!r}. "
        "Adicione o tratamento em transforms.py."
    )


def build_lenet_transform(
    preprocessing: dict[str, Any],
    *,
    training: bool,
    enable_augmentation: bool = True,
):
    """Converte a seção lenet_preprocessing do manifesto em transforms."""

    resize = preprocessing.get("resize", [32, 32])
    if not isinstance(resize, list) or len(resize) != 2:
        raise ValueError("lenet_preprocessing.resize deve ser [altura, largura].")
    image_size = (int(resize[0]), int(resize[1]))

    image_mode = str(preprocessing.get("image_mode", "grayscale")).lower()
    channels = int(preprocessing.get("channels", 1))

    pipeline: list[Any] = [transforms.Resize(image_size, antialias=True)]

    if image_mode == "grayscale" or channels == 1:
        pipeline.append(transforms.Grayscale(num_output_channels=1))
        in_channels = 1
    elif channels == 3:
        in_channels = 3
    else:
        raise ValueError(
            "A implementação aceita 1 canal (grayscale) ou 3 canais (RGB)."
        )

    if training and enable_augmentation:
        for item in preprocessing.get("train_augmentation", []):
            pipeline.append(_augmentation_from_manifest(item))

    pipeline.append(transforms.ToTensor())
    mean, std = _normalization(preprocessing)

    if len(mean) != in_channels:
        raise ValueError(
            f"A normalização possui {len(mean)} canal(is), mas a imagem usa "
            f"{in_channels}."
        )

    pipeline.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(pipeline), in_channels, image_size
