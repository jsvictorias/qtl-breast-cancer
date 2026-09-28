import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.loaders import build_loaders  # noqa: E402


@hydra.main(version_base="1.3", config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    train, val, tests, (mean, std) = build_loaders(cfg)
    x, y = next(iter(train))
    print(f"src={cfg.dataset.name} | mean={mean:.3f} std={std:.3f}")
    print(f"batch x={tuple(x.shape)} y={tuple(y.shape)}")
    for d, dl in tests.items():
        xb, _ = next(iter(dl))
        print(f"test[{d}] n={len(dl.dataset)} batch_mean={xb.mean():.3f}")  # type: ignore
    model = instantiate(cfg.model)
    logits = model(x)
    loss = nn.BCEWithLogitsLoss()(logits, y)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"logits={tuple(logits.shape)} params={n_params:,}")
    print(f"loss inicial={loss.item():.3f}")


if __name__ == "__main__":
    main()
