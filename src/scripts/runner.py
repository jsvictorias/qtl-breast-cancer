from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from classical.data.manifest_dataset import (
    ManifestImageDataset,
    load_processed_manifest,
)
from classical.models.lenet import LeNet, count_trainable_parameters
from classical.training.metrics import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_history,
    save_confusion_matrix_csv,
    save_metrics_json,
    save_predictions_csv,
)
from classical.training.train import fit_model, predict_model
from classical.transforms.transforms import build_lenet_transform
from classical.utils.config import deep_update, load_yaml
from classical.utils.reproducibility import set_global_seed


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path(__file__)).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    # Fallback caso pyproject.toml não seja localizado no topo
    for candidate in candidates:
        if (candidate / "src").is_dir():
            return candidate
    raise RuntimeError(
        "Não foi possível localizar a raiz do projeto. "
        "Garanta que o script esteja dentro da estrutura da pasta do projeto."
    )


PROJECT_ROOT = find_project_root()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def choose_device(requested: str) -> torch.device:
    requested = requested.lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada, mas não está disponível.")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS foi solicitado, mas não está disponível.")
    return device


def _worker_seed(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _class_weights(dataset: ManifestImageDataset, num_classes: int) -> torch.Tensor:
    counts = Counter(dataset.targets())
    total = len(dataset)
    weights = [
        total / (num_classes * counts[class_index])
        for class_index in range(num_classes)
    ]
    return torch.tensor(weights, dtype=torch.float32)


def _save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def get_default_config(project_root: Path) -> dict[str, Any]:
    """Retorna a configuração padrão voltada para o Small Dataset."""
    return {
        "dataset": {
            "manifest_path": "data/processed/manifest_processed.json",
            "batch_size": 16,
            "num_workers": 0,
            "pin_memory": True,
            "augmentation": True,
        },
        "model": {
            "conv1_channels": 6,
            "conv2_channels": 16,
            "fc1_units": 120,
            "fc2_units": 84,
            "dropout": 0.0,
        },
        "training": {
            "seed": 42,
            "device": "auto",
            "epochs": 30,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "use_class_weights": True,
            "scheduler_factor": 0.5,
            "scheduler_patience": 3,
            "minimum_learning_rate": 1e-6,
            "early_stopping_patience": 7,
            "early_stopping_min_delta": 1e-4,
        },
        "output": {
            "base_dir": "reports/outputs/small_dataset",
            "run_name": None,
        },
    }


def run_experiment(
    *,
    dataset_config_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    project_root: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    project_root = find_project_root(project_root)
    load_dotenv(project_root / ".env")

    configuration = get_default_config(project_root)

    # Carrega arquivos de configuração YAML caso existam
    if model_config_path:
        m_path = Path(model_config_path)
        if not m_path.is_absolute():
            m_path = project_root / m_path
        if m_path.is_file():
            configuration = deep_update(configuration, load_yaml(m_path))

    if dataset_config_path:
        d_path = Path(dataset_config_path)
        if not d_path.is_absolute():
            d_path = project_root / d_path
        if d_path.is_file():
            configuration = deep_update(configuration, load_yaml(d_path))

    if overrides:
        configuration = deep_update(configuration, overrides)

    dataset_config = configuration["dataset"]
    model_config = configuration["model"]
    training_config = configuration["training"]
    output_config = configuration["output"]

    seed = int(training_config.get("seed", 42))
    set_global_seed(seed)

    manifest_rel_path = dataset_config["manifest_path"]
    bundle = load_processed_manifest(
        manifest_path=manifest_rel_path,
        project_root=project_root,
    )

    preprocessing = bundle.manifest.get("lenet_preprocessing", {})
    train_transform, in_channels, image_size = build_lenet_transform(
        preprocessing,
        training=True,
        enable_augmentation=bool(dataset_config.get("augmentation", True)),
    )
    eval_transform, eval_channels, eval_size = build_lenet_transform(
        preprocessing,
        training=False,
        enable_augmentation=False,
    )
    if (in_channels, image_size) != (eval_channels, eval_size):
        raise RuntimeError("Transformações de treino e avaliação são incompatíveis.")

    train_dataset = ManifestImageDataset(bundle, "train", train_transform)
    val_dataset = ManifestImageDataset(bundle, "val", eval_transform)
    test_dataset = ManifestImageDataset(bundle, "test", eval_transform)

    batch_size = int(dataset_config.get("batch_size", 16))
    num_workers = int(dataset_config.get("num_workers", 0))
    requested_pin_memory = bool(dataset_config.get("pin_memory", True))

    device = choose_device(str(training_config.get("device", "auto")))
    pin_memory = requested_pin_memory and device.type == "cuda"

    generator = torch.Generator()
    generator.manual_seed(seed)

    common_loader_arguments = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "worker_init_fn": _worker_seed if num_workers > 0 else None,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common_loader_arguments,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_loader_arguments,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_arguments,
    )

    model = LeNet(
        in_channels=in_channels,
        num_classes=bundle.label_map.num_classes,
        conv1_channels=int(model_config.get("conv1_channels", 6)),
        conv2_channels=int(model_config.get("conv2_channels", 16)),
        fc1_units=int(model_config.get("fc1_units", 120)),
        fc2_units=int(model_config.get("fc2_units", 84)),
        dropout=float(model_config.get("dropout", 0.0)),
    ).to(device)

    use_class_weights = bool(training_config.get("use_class_weights", True))
    weights = (
        _class_weights(train_dataset, bundle.label_map.num_classes).to(device)
        if use_class_weights
        else None
    )
    criterion: nn.Module = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(training_config.get("scheduler_factor", 0.5)),
        patience=int(training_config.get("scheduler_patience", 3)),
        min_lr=float(training_config.get("minimum_learning_rate", 1e-6)),
    )

    base_output_dir = Path(output_config["base_dir"])
    if not base_output_dir.is_absolute():
        base_output_dir = project_root / base_output_dir

    run_name = output_config.get("run_name") or datetime.now().strftime(
        "run_%Y%m%d_%H%M%S"
    )
    output_dir = (base_output_dir / str(run_name)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_configuration = {
        **configuration,
        "resolved": {
            "project_root": str(project_root),
            "manifest_path": str(bundle.manifest_path),
            "dataset_root": str(bundle.dataset_root),
            "dataset_id": bundle.dataset_id,
            "device": str(device),
            "image_size": list(image_size),
            "in_channels": in_channels,
            "num_classes": bundle.label_map.num_classes,
            "class_to_index": bundle.label_map.to_dict(),
            "split_counts": bundle.split_counts(),
        },
    }
    _save_json(resolved_configuration, output_dir / "resolved_config.json")
    bundle.label_map.save(output_dir / "label_map.json")

    parameter_count = count_trainable_parameters(model)
    print("=" * 50)
    print(f" Executando LeNet - Dataset: {bundle.dataset_id}")
    print("=" * 50)
    print(f"Raiz do Projeto: {project_root}")
    print(f"Manifest: {bundle.manifest_path}")
    print(f"Classes: {bundle.label_map.to_dict()}")
    print(f"Amostras (Splits): {bundle.split_counts()}")
    print(f"Dimensão de Entrada: {in_channels}x{image_size[0]}x{image_size[1]}")
    print(f"Dispositivo de Execução: {device}")
    print(f"Parâmetros Treináveis: {parameter_count:,}")
    print(f"Diretório de Saída: {output_dir}\n")

    checkpoint_metadata = {
        "dataset_id": bundle.dataset_id,
        "class_to_index": bundle.label_map.to_dict(),
        "image_size": list(image_size),
        "in_channels": in_channels,
        "model_config": model_config,
    }

    history, best_epoch, best_val_loss = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=int(training_config.get("epochs", 30)),
        patience=int(training_config.get("early_stopping_patience", 7)),
        min_delta=float(training_config.get("early_stopping_min_delta", 1e-4)),
        checkpoint_path=output_dir / "best_model.pt",
        checkpoint_metadata=checkpoint_metadata,
    )

    _save_json(history, output_dir / "history.json")
    plot_history(history, output_dir)

    test_loss, targets, predictions, metadata = predict_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )
    metrics = compute_classification_metrics(
        targets=targets,
        predictions=predictions,
        label_map=bundle.label_map,
    )
    metrics.update(
        {
            "dataset_id": bundle.dataset_id,
            "evaluation_split": "test",
            "test_loss": test_loss,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "trainable_parameters": parameter_count,
            "device": str(device),
            "class_weights": (
                weights.detach().cpu().tolist() if weights is not None else None
            ),
        }
    )

    save_metrics_json(metrics, output_dir / "test_metrics.json")
    save_confusion_matrix_csv(
        metrics["confusion_matrix"],
        bundle.label_map,
        output_dir / "confusion_matrix.csv",
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        bundle.label_map,
        output_dir / "confusion_matrix.png",
    )
    save_predictions_csv(
        targets=targets,
        predictions=predictions,
        metadata=metadata,
        label_map=bundle.label_map,
        path=output_dir / "test_predictions.csv",
    )

    print("\n" + "-" * 40)
    print(" RESULTADOS DA AVALIAÇÃO DE TESTE")
    print("-" * 40)
    print(f"Loss Teste       : {test_loss:.4f}")
    print(f"Acurácia         : {metrics['accuracy']:.4f}")
    print(f"Macro Precision  : {metrics['macro_precision']:.4f}")
    print(f"Macro Recall     : {metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score   : {metrics['macro_f1']:.4f}")
    print("\nDesempenho por Classe:")
    for class_name, values in metrics["per_class"].items():
        print(
            f"  [{class_name}] "
            f"Precision: {values['precision']:.4f} | "
            f"Recall: {values['recall']:.4f} | "
            f"F1: {values['f1_score']:.4f} | "
            f"Suporte: {values['support']}"
        )

    return {
        "output_dir": output_dir,
        "history": history,
        "metrics": metrics,
        "model": model,
        "label_map": bundle.label_map,
        "configuration": resolved_configuration,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o experimento da LeNet.")
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="configs/dataset/small.yaml",
        help="Caminho opcional do YAML do dataset.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model/lenet.yaml",
        help="Caminho opcional do YAML do modelo.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--run-name")

    args = parser.parse_args()

    overrides = {}
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.device is not None:
        overrides.setdefault("training", {})["device"] = args.device
    if args.batch_size is not None:
        overrides.setdefault("dataset", {})["batch_size"] = args.batch_size
    if args.num_workers is not None:
        overrides.setdefault("dataset", {})["num_workers"] = args.num_workers
    if args.run_name is not None:
        overrides.setdefault("output", {})["run_name"] = args.run_name

    try:
        run_experiment(
            dataset_config_path=args.dataset_config,
            model_config_path=args.model_config,
            project_root=PROJECT_ROOT,
            overrides=overrides if overrides else None,
        )
    except Exception as error:
        print(f"\n[ERRO NA EXECUÇÃO]: {error}", file=sys.stderr)
        sys.exit(1)
