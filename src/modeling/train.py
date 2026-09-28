import copy
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn

from src.data_pipeline.loaders import build_loaders
from src.modeling.engine import compute_metrics, predict, train_one_epoch
from src.modeling.utils import compute_pos_weight, git_info, set_seed


@hydra.main(version_base="1.3", config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl, test_dls, (mean, std) = build_loaders(cfg)

    model = instantiate(cfg.model).to(device)
    pos_weight = compute_pos_weight(train_dl.dataset).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    best_auc, best_epoch, wait, best_state = -1.0, 0, 0, None
    for epoch in range(1, cfg.train.epochs + 1):
        loss = train_one_epoch(model, train_dl, criterion, optimizer, device)
        val = compute_metrics(*predict(model, val_dl, device))
        print(f"ep {epoch:02d} loss={loss:.3f} val_auc={val['roc_auc']:.3f}")

        if val["roc_auc"] > best_auc:
            best_auc, best_epoch, wait = val["roc_auc"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg.train.patience:
                break

    model.load_state_dict(best_state)
    print(f"best epoch={best_epoch} val_auc={best_auc:.3f}")

    out = Path(HydraConfig.get().runtime.output_dir)
    results = {
        "source": cfg.dataset.name,
        "seed": cfg.seed,
        **git_info(),
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "norm": {"mean": mean, "std": std},
        "pos_weight": pos_weight.item(),
        "targets": {},
    }
    for domain, dl in test_dls.items():
        probs, labels = predict(model, dl, device)
        results["targets"][domain] = compute_metrics(probs, labels, cfg.train.threshold)
        np.savez(out / f"preds_{domain}.npz", probs=probs, labels=labels)

    torch.save(best_state, out / "model.pt")
    (out / "metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
