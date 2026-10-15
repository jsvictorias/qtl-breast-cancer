import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from src.data_pipeline.ultrasound_dataset import UltrasoundDataset


def compute_mean_std(dataset, batch_size=64):
    loader = DataLoader(dataset, batch_size=batch_size)
    s, s2, n = 0.0, 0.0, 0
    for x, _ in loader:
        s += x.sum().item()
        s2 += (x**2).sum().item()
        n += x.numel()
    mean = s / n
    return mean, (s2 / n - mean**2) ** 0.5


def build_loaders(cfg):
    images_path = cfg.data.images_path
    meta_path = cfg.data.meta_path
    src = cfg.dataset.name

    def make(domain, split, shuffle=False):
        ds = UltrasoundDataset(images_path, meta_path, domain, split, tf)
        return DataLoader(
            ds,
            batch_size=cfg.train.batch_size,
            shuffle=shuffle,
            num_workers=cfg.data.num_workers,
            generator=g,
        )

    fraction = cfg.train.get("train_fraction", 1.0)
    stats_ds = UltrasoundDataset(images_path, meta_path, src, "train")
    stats_ds.subsample(fraction, cfg.seed)
    mean, std = compute_mean_std(stats_ds)

    tf = v2.Normalize([mean], [std])
    g = torch.Generator().manual_seed(cfg.seed)

    train = make(src, "train", shuffle=True)
    train.dataset.subsample(fraction, cfg.seed)  # type: ignore
    val = make(src, "val")
    tests = {d: make(d, "test") for d in cfg.eval_domains}
    vals = {d: make(d, "val") for d in cfg.eval_domains}

    return train, val, tests, (mean, std), vals
