import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class UltrasoundDataset(Dataset):
    def __init__(self, manifest_path, domain, split, transform=None):
        manifest = json.loads(
            (PROJECT_ROOT / manifest_path).read_text(encoding="utf-8")
        )
        self.samples = [
            s
            for s in manifest["samples"]
            if s["dataset"] == domain and s["split"] == split
        ]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(PROJECT_ROOT / s["image_path"]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(s["label_id"], dtype=torch.float32)
