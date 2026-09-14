from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

BAND_FRACTION = 0.08
NEAR_BLACK_MEAN = 12.0
TEXT_EDGE_RATIO = 1.3
TEXT_MIN_BAND_MEAN = 40.0

ROI_MARGIN = 0.08

CHROMA_MOD_THRESHOLD = 15
CHROMA_HIGH_THRESHOLD = 60

SPECKLE_BLOCK = 8


def to_grayscale_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float64)


def _gradient_energy(band: np.ndarray) -> float:
    if band.size == 0:
        return 0.0
    gy = np.abs(np.diff(band, axis=0)).mean() if band.shape[0] > 1 else 0.0
    gx = np.abs(np.diff(band, axis=1)).mean() if band.shape[1] > 1 else 0.0
    return float(gy + gx)


@dataclass
class BandDiagnostics:
    top_mean: float
    bottom_mean: float
    top_near_black: bool
    bottom_near_black: bool
    top_text_candidate: bool
    bottom_text_candidate: bool


def band_diagnostics(
    gray: np.ndarray, band_fraction: float = BAND_FRACTION
) -> BandDiagnostics:
    h, _w = gray.shape
    band_h = max(1, round(h * band_fraction))
    if h <= 2 * band_h:
        band_h = max(1, h // 3)
    top, bottom = gray[:band_h, :], gray[-band_h:, :]
    mid = gray[band_h:-band_h, :] if h > 2 * band_h else gray

    mid_edge = _gradient_energy(mid)
    top_edge = _gradient_energy(top)
    bottom_edge = _gradient_energy(bottom)
    top_mean, bottom_mean = float(top.mean()), float(bottom.mean())

    def is_text(band_mean: float, band_edge: float) -> bool:
        denom = mid_edge if mid_edge > 1e-6 else 1e-6
        return band_mean > TEXT_MIN_BAND_MEAN and (band_edge / denom) > TEXT_EDGE_RATIO

    return BandDiagnostics(
        top_mean=top_mean,
        bottom_mean=bottom_mean,
        top_near_black=top_mean < NEAR_BLACK_MEAN,
        bottom_near_black=bottom_mean < NEAR_BLACK_MEAN,
        top_text_candidate=is_text(top_mean, top_edge),
        bottom_text_candidate=is_text(bottom_mean, bottom_edge),
    )


def central_roi(gray: np.ndarray, margin: float = ROI_MARGIN) -> np.ndarray:
    h, w = gray.shape
    mh, mw = round(h * margin), round(w * margin)
    if h > 2 * mh and w > 2 * mw:
        roi = gray[mh : h - mh, mw : w - mw]
    else:
        roi = gray
    return roi if roi.size else gray


@dataclass
class IntensityStats:
    mean: float
    std: float
    median: float
    p01: float
    p99: float
    skewness: float
    kurtosis: float
    entropy_bits: float


def intensity_stats(roi: np.ndarray) -> IntensityStats:
    flat = roi.ravel()
    mean, std = float(flat.mean()), float(flat.std())
    if std > 1e-6:
        centered = flat - mean
        skew = float(np.mean(centered**3) / std**3)
        kurt = float(np.mean(centered**4) / std**4 - 3.0)
    else:
        skew, kurt = 0.0, 0.0
    hist, _ = np.histogram(flat, bins=256, range=(0, 255))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    return IntensityStats(
        mean=mean,
        std=std,
        median=float(np.median(flat)),
        p01=float(np.percentile(flat, 1)),
        p99=float(np.percentile(flat, 99)),
        skewness=skew,
        kurtosis=kurt,
        entropy_bits=entropy,
    )


def snr_db(roi: np.ndarray, foreground_threshold: float = 5.0) -> float:
    fg = roi[roi > foreground_threshold]
    if fg.size < 16:
        fg = roi.ravel()
    mean, std = float(fg.mean()), float(fg.std())
    if mean <= 1e-6 or std <= 1e-6:
        return float("nan")
    return float(20.0 * np.log10(mean / std))


def speckle_index(roi: np.ndarray, block: int = SPECKLE_BLOCK) -> float:
    h, w = roi.shape
    h2, w2 = (h // block) * block, (w // block) * block
    if h2 == 0 or w2 == 0:
        return float("nan")
    cropped = roi[:h2, :w2]
    blocks = cropped.reshape(h2 // block, block, w2 // block, block).swapaxes(1, 2)
    means = blocks.mean(axis=(2, 3))
    stds = blocks.std(axis=(2, 3))
    valid = means > 5.0
    if not np.any(valid):
        return float("nan")
    cv = stds[valid] / means[valid]
    return float(np.median(cv))


@dataclass
class ChromaStats:
    frac_moderate: float
    frac_high: float
    max_channel_diff: float


def chroma_stats(
    rgb: np.ndarray,
    moderate: float = CHROMA_MOD_THRESHOLD,
    high: float = CHROMA_HIGH_THRESHOLD,
) -> ChromaStats:
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return ChromaStats(0.0, 0.0, 0.0)
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    return ChromaStats(
        frac_moderate=float((chroma > moderate).mean()),
        frac_high=float((chroma > high).mean()),
        max_channel_diff=float(chroma.max()),
    )


def extract_image_features(image_path: Path) -> dict[str, Any]:
    with Image.open(image_path) as image:
        fmt = image.format
        mode = image.mode
        width, height = image.size
        raw = np.asarray(image)
        gray = to_grayscale_array(image)

    bit_depth = raw.dtype.itemsize * 8
    channels = 1 if raw.ndim == 2 else raw.shape[2]

    bands = band_diagnostics(gray)
    roi = central_roi(gray)
    stats = intensity_stats(roi)
    chroma = chroma_stats(raw) if raw.ndim == 3 else ChromaStats(0.0, 0.0, 0.0)

    return {
        "width": width,
        "height": height,
        "aspect_ratio": width / height if height else float("nan"),
        "n_pixels": width * height,
        "format": fmt,
        "mode": mode,
        "channels": channels,
        "bit_depth": bit_depth,
        "file_size_bytes": image_path.stat().st_size,
        "roi_mean": stats.mean,
        "roi_std": stats.std,
        "roi_median": stats.median,
        "roi_p01": stats.p01,
        "roi_p99": stats.p99,
        "roi_skewness": stats.skewness,
        "roi_kurtosis": stats.kurtosis,
        "roi_entropy_bits": stats.entropy_bits,
        "snr_db": snr_db(roi),
        "speckle_index": speckle_index(roi),
        "chroma_frac_moderate": chroma.frac_moderate,
        "chroma_frac_high": chroma.frac_high,
        "chroma_max_channel_diff": chroma.max_channel_diff,
        "top_band_mean": bands.top_mean,
        "bottom_band_mean": bands.bottom_mean,
        "top_band_near_black": bands.top_near_black,
        "bottom_band_near_black": bands.bottom_near_black,
        "top_band_text_candidate": bands.top_text_candidate,
        "bottom_band_text_candidate": bands.bottom_text_candidate,
    }
