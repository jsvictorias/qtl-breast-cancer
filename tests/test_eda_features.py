import numpy as np

from src.data_pipeline.eda_features import (
    band_diagnostics,
    central_roi,
    chroma_stats,
    intensity_stats,
    snr_db,
    speckle_index,
)


def test_band_diagnostics_flags_near_black_bottom_border() -> None:
    gray = np.full((200, 200), 120.0)
    gray[-16:, :] = 2.0

    bands = band_diagnostics(gray)

    assert bands.bottom_near_black
    assert not bands.top_near_black
    assert not bands.bottom_text_candidate


def test_band_diagnostics_flags_bright_high_edge_top_band_as_text_candidate() -> None:
    rng = np.random.default_rng(0)
    gray = np.full((200, 200), 60.0)
    header = np.zeros((16, 200))
    header[:, ::2] = 220.0
    header[:, 1::2] = 60.0
    gray[:16, :] = header
    gray[16:] += rng.normal(0, 1.0, size=gray[16:].shape)

    bands = band_diagnostics(gray)

    assert bands.top_text_candidate
    assert not bands.top_near_black


def test_central_roi_excludes_outer_margin() -> None:
    gray = np.zeros((100, 100))
    gray[8:92, 8:92] = 1.0

    roi = central_roi(gray, margin=0.08)

    assert roi.min() == 1.0
    assert roi.shape[0] < gray.shape[0]
    assert roi.shape[1] < gray.shape[1]


def test_intensity_stats_matches_known_moments() -> None:
    flat = np.array([10.0, 10.0, 10.0, 10.0])
    stats = intensity_stats(flat.reshape(2, 2))

    assert stats.mean == 10.0
    assert stats.std == 0.0
    assert stats.skewness == 0.0
    assert stats.kurtosis == 0.0


def test_snr_db_is_higher_for_a_cleaner_image() -> None:
    rng = np.random.default_rng(1)
    base = np.full((64, 64), 100.0)
    clean = base + rng.normal(0, 2.0, base.shape)
    noisy = base + rng.normal(0, 25.0, base.shape)

    assert snr_db(clean) > snr_db(noisy)


def test_snr_db_returns_nan_for_an_all_black_roi() -> None:
    roi = np.zeros((32, 32))

    assert np.isnan(snr_db(roi))


def test_speckle_index_is_higher_for_locally_noisy_texture() -> None:
    rng = np.random.default_rng(2)
    base = np.full((64, 64), 100.0)
    smooth = base + rng.normal(0, 0.5, base.shape)
    speckled = base + rng.normal(0, 20.0, base.shape)

    assert speckle_index(speckled) > speckle_index(smooth)


def test_speckle_index_ignores_near_black_blocks() -> None:
    roi = np.zeros((32, 32))

    assert np.isnan(speckle_index(roi, block=8))


def test_chroma_stats_is_zero_for_a_true_gray_image() -> None:
    gray_value = np.full((10, 10), 128, dtype=np.uint8)
    rgb = np.stack([gray_value, gray_value, gray_value], axis=-1)

    stats = chroma_stats(rgb)

    assert stats.frac_moderate == 0.0
    assert stats.frac_high == 0.0
    assert stats.max_channel_diff == 0.0


def test_chroma_stats_flags_a_localized_saturated_marker_but_not_mild_tint() -> None:
    rgb = np.full((10, 10, 3), 100, dtype=np.uint8)
    rgb[..., 0] += 20
    rgb[0, 0] = [255, 0, 0]

    stats = chroma_stats(rgb)

    assert stats.frac_moderate > 0.9
    assert stats.frac_high == 0.01
    assert stats.max_channel_diff == 255.0


def test_chroma_stats_on_grayscale_mode_array_is_zero() -> None:
    stats = chroma_stats(np.full((10, 10), 128, dtype=np.uint8))

    assert stats == chroma_stats(np.full((10, 10), 128, dtype=np.uint8))
    assert stats.frac_moderate == 0.0
