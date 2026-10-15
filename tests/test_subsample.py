import numpy as np

from src.data_pipeline.ultrasound_dataset import UltrasoundDataset


def _fake(n_neg=70, n_pos=30):
    ds = UltrasoundDataset.__new__(UltrasoundDataset)
    labels = [0] * n_neg + [1] * n_pos
    ds.images = np.arange(len(labels))[:, None, None].repeat(2, 1).repeat(2, 2)
    ds.samples = [{"sample_id": f"s{i}", "label_id": y} for i, y in enumerate(labels)]
    return ds


def test_subsample_is_stratified_and_aligned():
    ds = _fake()
    ds.subsample(0.25, seed=3)
    labels = [s["label_id"] for s in ds.samples]
    # round(70*.25)=18 negatives, round(30*.25)=8 positives
    assert labels.count(0) == 18 and labels.count(1) == 8
    ids = [int(s["sample_id"][1:]) for s in ds.samples]
    assert (ds.images[:, 0, 0] == np.array(ids)).all()  # images follow samples


def test_subsample_deterministic_per_seed_and_full_is_noop():
    a, b, c = _fake(), _fake(), _fake()
    a.subsample(0.5, 1)
    b.subsample(0.5, 1)
    c.subsample(0.5, 2)
    assert [s["sample_id"] for s in a.samples] == [s["sample_id"] for s in b.samples]
    assert [s["sample_id"] for s in a.samples] != [s["sample_id"] for s in c.samples]
    d = _fake()
    d.subsample(1.0, 0)
    assert len(d) == 100
