"""Tests that turn my no-leakage claims into something checkable.

EN: A README can claim "no leakage" and be wrong. These tests assert the three
    guarantees mechanically: the split is chronological, the scaler saw only
    train, and no input window crosses a split boundary. They run on a small
    slice of the CSV so the whole file finishes in seconds.
TR: Bir README "sızıntı yok" diyebilir ve yanılıyor olabilir. Bu testler üç
    garantiyi mekanik olarak doğruluyor: bölme kronolojik, ölçekleyici yalnızca
    train'i gördü ve hiçbir girdi penceresi bölme sınırını aşmıyor. CSV'nin
    küçük bir diliminde koşuyorlar, dolayısıyla dosyanın tamamı saniyeler sürüyor.

Run / Koşum:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.baselines import persistence_predictions
from src.data import build_datasets, chronological_split, clean, load_raw, to_hourly

# EN: ~2 years of 10-minute data: enough to be realistic, fast enough for CI.
# TR: ~2 yıllık 10 dakikalık veri: gerçekçi olacak kadar çok, CI için yeterince hızlı.
TEST_NROWS = 100_000
LOOKBACK = 48
HORIZON = 12


@pytest.fixture(scope="module")
def bundle():
    return build_datasets(nrows=TEST_NROWS, lookback=LOOKBACK, horizon=HORIZON)


@pytest.fixture(scope="module")
def hourly():
    cleaned, _ = clean(load_raw(nrows=TEST_NROWS))
    return to_hourly(cleaned)


# --------------------------------------------------------------------------- #
# Cleaning / Temizleme
# --------------------------------------------------------------------------- #
def test_sentinels_are_gone(hourly):
    """No -9999 survives cleaning / Temizlikten sonra -9999 kalmıyor."""
    for col in config.SENTINEL_COLS:
        assert (hourly[col] > config.SENTINEL_VALUE).all(), col


def test_no_nans_after_cleaning(hourly):
    assert not hourly.isna().any().any()


def test_grid_is_exactly_hourly(hourly):
    """Every consecutive pair is one hour apart / Her ardışık çift bir saat arayla."""
    deltas = hourly.index.to_series().diff().dropna().unique()
    assert list(deltas) == [pd.Timedelta(hours=1)]


def test_timestamps_are_unique_and_sorted(hourly):
    assert hourly.index.is_unique
    assert hourly.index.is_monotonic_increasing


# --------------------------------------------------------------------------- #
# Split / Bölme
# --------------------------------------------------------------------------- #
def test_split_is_chronological(hourly):
    """Train is strictly older than val, val strictly older than test."""
    train, val, test = chronological_split(hourly)
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()


def test_split_covers_everything_without_overlap(hourly):
    train, val, test = chronological_split(hourly)
    assert len(train) + len(val) + len(test) == len(hourly)
    assert set(train.index).isdisjoint(val.index)
    assert set(val.index).isdisjoint(test.index)


# --------------------------------------------------------------------------- #
# Normalisation / Normalizasyon
# --------------------------------------------------------------------------- #
def test_scaler_statistics_come_from_train_only(bundle, hourly):
    """The core anti-leakage guarantee / Sızıntıya karşı ana garanti."""
    train, _, _ = chronological_split(hourly)

    expected_mean = train[config.TARGET_COL].mean()
    expected_std = train[config.TARGET_COL].std()

    assert bundle.normalizer.target_mean == pytest.approx(expected_mean, rel=1e-9)
    assert bundle.normalizer.target_std == pytest.approx(expected_std, rel=1e-9)

    # EN: and it must NOT equal the whole-set statistics, or nothing was proven.
    # TR: ve tüm verinin istatistiğine EŞİT OLMAMALI, yoksa hiçbir şey kanıtlanmaz.
    whole_mean = hourly[config.TARGET_COL].mean()
    assert bundle.normalizer.target_mean != pytest.approx(whole_mean, rel=1e-6)


def test_train_features_are_standardised(bundle):
    """Train features should be ~N(0, 1) after scaling."""
    feats = bundle.train.features
    assert abs(float(feats.mean())) < 0.05
    assert 0.8 < float(feats.std()) < 1.2


# --------------------------------------------------------------------------- #
# Windowing / Pencereleme
# --------------------------------------------------------------------------- #
def test_no_window_crosses_a_split_boundary(bundle):
    """Each dataset's windows live entirely inside their own split.

    EN: Because I window each split separately, the timestamps a dataset can
        possibly touch are bounded by that split's own first and last stamp.
    TR: Her bölmeyi ayrı pencerelediğim için, bir veri setinin dokunabileceği
        zaman damgaları o bölmenin kendi ilk ve son damgasıyla sınırlı.
    """
    train_end = bundle.train.timestamps.max()
    val_start = bundle.val.timestamps.min()
    val_end = bundle.val.timestamps.max()
    test_start = bundle.test.timestamps.min()

    assert train_end < val_start
    assert val_end < test_start

    for name in ("train", "val", "test"):
        ds = getattr(bundle, name)
        # EN: the earliest timestep any window reads, and the latest target.
        # TR: herhangi bir pencerenin okuduğu en erken adım ve en geç hedef.
        assert ds.target_times.min() >= ds.timestamps.min()
        assert ds.target_times.max() <= ds.timestamps.max()


def test_sample_count_matches_the_formula(bundle):
    """Kept + dropped windows must account for every candidate position."""
    for name in ("train", "val", "test"):
        ds = getattr(bundle, name)
        candidates = len(ds.features) - ds.lookback - ds.horizon + 1
        assert len(ds) + ds.n_dropped == candidates


def test_no_window_touches_interpolated_data(hourly):
    """Windows over invented hours are dropped, not scored.

    EN: The raw series has a 74-hour gap in October 2016 that lands inside the
        test slice. Interpolating it invents weather, and scoring a model on
        invented targets is not an honest test, so those windows are excluded.
    TR: Ham seride Ekim 2016'da 74 saatlik bir boşluk var ve tam test diliminin
        içine düşüyor. Onu interpole etmek hava durumu uydurmaktır; uydurulmuş
        hedefler üzerinden puanlamak dürüst bir sınav olmadığından o pencereler
        eleniyor.
    """
    from src.data import chronological_split

    synthetic = hourly[config.SYNTHETIC_COL].astype(bool)
    b = build_datasets(nrows=TEST_NROWS, lookback=LOOKBACK, horizon=HORIZON)

    for name in ("train", "val", "test"):
        ds = getattr(b, name)
        flags = synthetic.loc[ds.timestamps].to_numpy()
        for i in range(0, len(ds), max(1, len(ds) // 200)):
            start = int(ds.sample_index[i])
            assert not flags[start : start + ds.lookback].any()
            assert not flags[start + ds.lookback - 1 + ds.horizon]


def test_interpolated_windows_can_be_kept_on_request(hourly):
    """The exclusion is a choice I can turn off and measure."""
    strict = build_datasets(nrows=TEST_NROWS, lookback=LOOKBACK, horizon=HORIZON)
    loose = build_datasets(
        nrows=TEST_NROWS,
        lookback=LOOKBACK,
        horizon=HORIZON,
        exclude_interpolated=False,
    )
    assert len(loose.test) >= len(strict.test)
    assert loose.test.n_dropped == 0


def test_target_is_exactly_horizon_hours_after_the_window(bundle):
    """The label really is H hours after the last observed timestep."""
    ds = bundle.val
    for i in (0, 1, len(ds) // 2, len(ds) - 1):
        start = int(ds.sample_index[i])
        last_observed_time = ds.timestamps[start + ds.lookback - 1]
        target_time = ds.target_times[i]
        assert target_time - last_observed_time == pd.Timedelta(hours=ds.horizon)


def test_input_window_never_contains_the_target(bundle):
    """The model cannot see its own answer / Model kendi cevabını göremiyor."""
    ds = bundle.val
    for i in (0, len(ds) // 3, len(ds) - 1):
        start = int(ds.sample_index[i])
        window_times = ds.timestamps[start : start + ds.lookback]
        assert ds.target_times[i] not in window_times


def test_getitem_matches_the_celsius_accessors(bundle):
    """__getitem__ and targets_celsius must describe the same sample."""
    ds = bundle.test
    for i in (0, 7, len(ds) - 1):
        _, y = ds[i]
        restored = bundle.normalizer.inverse_transform_target(np.array([y.item()]))[0]
        assert restored == pytest.approx(float(ds.targets_celsius[i]), abs=1e-3)


def test_persistence_uses_the_last_observed_value(bundle):
    """The baseline predicts the window's final temperature, nothing else."""
    ds = bundle.test
    preds = persistence_predictions(ds)
    assert len(preds) == len(ds)
    for i in (0, 5, len(ds) // 2, len(ds) - 1):
        # EN: go through sample_index - after dropping interpolated windows the
        #     i-th sample is no longer the i-th position in the slice.
        # TR: sample_index üzerinden git - interpole pencereler elendikten sonra
        #     i. örnek artık dilimdeki i. konum değil.
        start = int(ds.sample_index[i])
        expected = ds.target_celsius[start + ds.lookback - 1]
        assert preds[i] == pytest.approx(float(expected), abs=1e-5)
        # EN: and it must be exactly `horizon` hours before the target it scores.
        # TR: ve puanladığı hedeften tam `horizon` saat önce olmalı.
        assert ds.target_times[i] - ds.timestamps[start + ds.lookback - 1] == pd.Timedelta(
            hours=ds.horizon
        )


def test_shapes_are_what_the_models_expect(bundle):
    x, y = bundle.train[0]
    assert tuple(x.shape) == (LOOKBACK, bundle.n_features)
    assert tuple(y.shape) == ()


# --------------------------------------------------------------------------- #
# Feature subsets / Öznitelik alt kümeleri
# --------------------------------------------------------------------------- #
def test_univariate_subset_keeps_only_temperature():
    b = build_datasets(
        nrows=TEST_NROWS,
        lookback=LOOKBACK,
        horizon=HORIZON,
        feature_subset=[config.TARGET_COL],
    )
    assert b.n_features == 1
    assert b.feature_names == [config.TARGET_COL]


def test_dropping_the_target_is_rejected():
    with pytest.raises(KeyError):
        build_datasets(nrows=TEST_NROWS, feature_subset=["p (mbar)"])


def test_engineered_features_are_added():
    b = build_datasets(nrows=TEST_NROWS, lookback=LOOKBACK, horizon=HORIZON,
                       use_engineered_features=True)
    for col in ("Wx", "Wy", "day sin", "day cos", "year sin", "year cos"):
        assert col in b.feature_names
    assert "wd (deg)" not in b.feature_names
