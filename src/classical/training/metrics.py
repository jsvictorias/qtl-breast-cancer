from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from classical.utils.label_map import LabelMap


def confusion_matrix(
    targets: list[int],
    predictions: list[int],
    num_classes: int,
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions, strict=True):
        matrix[target, prediction] += 1
    return matrix


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_classification_metrics(
    *,
    targets: list[int],
    predictions: list[int],
    label_map: LabelMap,
) -> dict[str, Any]:
    matrix = confusion_matrix(targets, predictions, label_map.num_classes)
    total = int(matrix.sum())
    correct = int(np.trace(matrix))

    per_class: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    weighted_f1_sum = 0.0

    for class_index, class_name in label_map.index_to_class.items():
        true_positive = int(matrix[class_index, class_index])
        false_positive = int(matrix[:, class_index].sum() - true_positive)
        false_negative = int(matrix[class_index, :].sum() - true_positive)
        support = int(matrix[class_index, :].sum())

        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2 * precision * recall, precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        weighted_f1_sum += f1 * support

        per_class[class_name] = {
            "index": class_index,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }

    return {
        "accuracy": _safe_divide(correct, total),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "weighted_f1": _safe_divide(weighted_f1_sum, total),
        "total_samples": total,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "class_order": label_map.classes,
    }


def save_metrics_json(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def save_confusion_matrix_csv(
    matrix: list[list[int]],
    label_map: LabelMap,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\predicted", *label_map.classes])
        for class_name, row in zip(label_map.classes, matrix, strict=True):
            writer.writerow([class_name, *row])


def save_predictions_csv(
    *,
    targets: list[int],
    predictions: list[int],
    metadata: list[dict[str, str]],
    label_map: LabelMap,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_id",
                "path",
                "true_index",
                "true_label",
                "predicted_index",
                "predicted_label",
                "correct",
            ],
        )
        writer.writeheader()

        for target, prediction, item in zip(
            targets, predictions, metadata, strict=True
        ):
            writer.writerow(
                {
                    "sample_id": item["sample_id"],
                    "path": item["path"],
                    "true_index": target,
                    "true_label": label_map.decode(target),
                    "predicted_index": prediction,
                    "predicted_label": label_map.decode(prediction),
                    "correct": target == prediction,
                }
            )


def plot_confusion_matrix(
    matrix: list[list[int]],
    label_map: LabelMap,
    path: Path,
) -> None:
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(values)
    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(label_map.num_classes),
        yticks=np.arange(label_map.num_classes),
        xticklabels=label_map.classes,
        yticklabels=label_map.classes,
        xlabel="Classe predita",
        ylabel="Classe real",
        title="Matriz de confusão",
    )

    threshold = values.max() / 2 if values.size else 0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_curve(
    *,
    epochs: list[int],
    train_values: list[float],
    val_values: list[float],
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(epochs, train_values, label="Treino")
    axis.plot(epochs, val_values, label="Validação")
    axis.set_xlabel("Época")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend()
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_history(history: dict[str, list[float]], output_dir: Path) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    _plot_curve(
        epochs=epochs,
        train_values=history["train_loss"],
        val_values=history["val_loss"],
        ylabel="Loss",
        title="Loss por época",
        path=output_dir / "loss_curve.png",
    )
    _plot_curve(
        epochs=epochs,
        train_values=history["train_accuracy"],
        val_values=history["val_accuracy"],
        ylabel="Acurácia",
        title="Acurácia por época",
        path=output_dir / "accuracy_curve.png",
    )
