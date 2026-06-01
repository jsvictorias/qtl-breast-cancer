import os
from pathlib import Path


def test_check_cluster_datasets():
    base_dir = os.getenv("DATASET_DIR")
    assert base_dir is not None, (
        "A variável DATASET_DIR não está configurada no ambiente."
    )

    dataset_path = Path(base_dir)

    large_dataset = dataset_path / "large-dataset" / "ultrasound breast classification"
    medium_dataset = dataset_path / "medium-dataset" / "Dataset_BUSI_with_GT"
    small_dataset = dataset_path / "small-dataset" / "small-dataset-256"

    print(f"\n[INFO] Validando datasets em: {dataset_path.resolve()}")

    assert small_dataset.exists(), f"Small dataset não encontrado em: {small_dataset}"
    assert medium_dataset.exists(), (
        f"Medium dataset não encontrado em: {medium_dataset}"
    )
    assert large_dataset.exists(), f"Large dataset não encontrado em: {large_dataset}"

    print(
        "√ Todos os datasets do cluster de homologação foram localizados com sucesso!"
    )


def test_qml_dummy_pipeline():
    pass
