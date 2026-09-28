import random
import subprocess

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_pos_weight(dataset) -> torch.Tensor:
    labels = torch.tensor([s["label_id"] for s in dataset.samples])
    n_pos = labels.sum()
    return (len(labels) - n_pos) / n_pos


def git_info() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
    except Exception:
        commit, dirty = "unknown", None
    return {"commit": commit, "dirty": dirty}
