import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from src.data_pipeline.ultrasound_dataset import UltrasoundDataset


def base_transform(size):
    return v2.Compose(
        [
            v2.Resize((size, size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


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
    size, mp = cfg.data.image_size, cfg.data.manifest_path
    src = cfg.dataset.name

    stats_ds = UltrasoundDataset(mp, src, "train", base_transform(size))
    mean, std = compute_mean_std(stats_ds)

    tf = v2.Compose([base_transform(size), v2.Normalize([mean], [std])])
    g = torch.Generator().manual_seed(cfg.seed)

    def make(domain, split, shuffle=False):
        ds = UltrasoundDataset(mp, domain, split, tf)
        return DataLoader(
            ds,
            batch_size=cfg.train.batch_size,
            shuffle=shuffle,
            num_workers=cfg.data.num_workers,
            generator=g,
        )

    train = make(src, "train", shuffle=True)
    val = make(src, "val")
    tests = {d: make(d, "test") for d in cfg.eval_domains}
    return train, val, tests, (mean, std)
