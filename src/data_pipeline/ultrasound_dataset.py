from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class UltrasoundDataset(Dataset):
    def __init__(self, images_path, meta_path, domain, split, transform=None):
        meta = pd.read_csv(PROJECT_ROOT / meta_path)
        images = np.load(PROJECT_ROOT / images_path)["images"]
        rows = meta.index[(meta["dataset"] == domain) & (meta["split"] == split)]
        self.images = images[rows]
        self.samples = meta.loc[rows, ["sample_id", "label_id"]].to_dict("records")
        self.transform = transform

    def subsample(self, fraction: float, seed: int) -> None:
        """Keep a label-stratified random `fraction` of the samples (in place)."""
        if fraction >= 1.0:
            return
        rng = np.random.default_rng(seed)
        labels = np.array([s["label_id"] for s in self.samples])
        keep = []
        for lab in np.unique(labels):
            idx = np.flatnonzero(labels == lab)
            n = max(1, round(len(idx) * fraction))
            keep.append(rng.choice(idx, size=n, replace=False))
        keep = np.sort(np.concatenate(keep))
        self.images = self.images[keep]
        self.samples = [self.samples[i] for i in keep]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).float().div(255).unsqueeze(0)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.samples[idx]["label_id"], dtype=torch.float32)
