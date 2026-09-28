import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

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


if __name__ == "__main__":
    main()
