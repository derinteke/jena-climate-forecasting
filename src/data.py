"""Download, clean, split, normalise and window the Jena climate data.

EN: This module is where every anti-leakage decision is enforced. Three rules
    are non-negotiable here:
      1. The train/val/test split is chronological, never random.
      2. Mean/std come from the TRAIN slice only and are then applied to all.
      3. I split FIRST and window each slice SEPARATELY, so no input window ever
         spans a split boundary. Many tutorials window first and split after,
         which silently leaks train timesteps into validation windows.
TR: Sızıntıya karşı bütün kararlar bu modülde uygulanıyor. Üç kural pazarlıksız:
      1. train/val/test bölmesi kronolojik, asla rastgele değil.
      2. mean/std YALNIZCA train diliminden hesaplanıp hepsine uygulanıyor.
      3. Önce BÖLÜP sonra her dilimi AYRI AYRI pencereliyorum; böylece hiçbir
         girdi penceresi bölme sınırını aşmıyor. Çoğu eğitim materyali önce
         pencereleyip sonra bölüyor; bu da train adımlarını sessizce validation
         penceresine sızdırıyor.
"""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src import config


# --------------------------------------------------------------------------- #
# 1. Download and load / İndirme ve okuma
# --------------------------------------------------------------------------- #
def download_dataset(force: bool = False) -> Path:
    """Fetch and unzip the CSV if it is not already on disk.

    EN: Keras' get_file does the same job, but I keep this framework-free so the
        project does not need TensorFlow installed just to download a file.
    TR: Keras'ın get_file fonksiyonu da aynı işi yapıyor ama sırf dosya indirmek
        için projeye TensorFlow bağımlılığı eklemeyeyim diye elle yazdım.
    """
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    if config.CSV_PATH.exists() and not force:
        return config.CSV_PATH

    if not config.ZIP_PATH.exists() or force:
        urllib.request.urlretrieve(config.DATASET_URL, config.ZIP_PATH)

    with zipfile.ZipFile(config.ZIP_PATH) as zf:
        zf.extractall(config.RAW_DIR)

    if not config.CSV_PATH.exists():
        raise FileNotFoundError(
            f"Expected {config.CSV_PATH} after extracting {config.ZIP_PATH}."
        )
    return config.CSV_PATH


def load_raw(nrows: int | None = None) -> pd.DataFrame:
    """Read the CSV with a parsed, sorted DatetimeIndex."""
    path = download_dataset()
    df = pd.read_csv(path, nrows=nrows)
    df[config.TIME_COL] = pd.to_datetime(
        df[config.TIME_COL], format=config.TIME_FORMAT
    )
    df = df.set_index(config.TIME_COL).sort_index()
    return df


# --------------------------------------------------------------------------- #
# 2. Cleaning / Temizleme
# --------------------------------------------------------------------------- #
@dataclass
class CleaningReport:
    """What the cleaning step actually changed / Temizliğin gerçekte değiştirdikleri."""

    rows_in: int = 0
    rows_out: int = 0
    duplicate_timestamps: int = 0
    sentinel_cells: dict = field(default_factory=dict)
    missing_steps_filled: int = 0
    longest_gap_steps: int = 0

    def summary(self) -> str:
        gap_hours = self.longest_gap_steps * config.RAW_STEP_MINUTES / 60
        return "\n".join(
            [
                f"rows in .................. {self.rows_in:,}",
                f"rows out ................. {self.rows_out:,}",
                f"duplicate timestamps ..... {self.duplicate_timestamps:,} (dropped)",
                f"sentinel (-9999) cells ... {self.sentinel_cells} (interpolated)",
                f"missing grid steps ....... {self.missing_steps_filled:,} (interpolated)",
                f"longest contiguous gap ... {self.longest_gap_steps:,} steps "
                f"(~{gap_hours:.1f} h)",
            ]
        )


def _longest_nan_run(s: pd.Series) -> int:
    """Length of the longest consecutive NaN run / En uzun ardışık NaN dizisi."""
    isna = s.isna().to_numpy()
    if not isna.any():
        return 0
    longest = run = 0
    for flag in isna:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return longest


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Fix the two real defects in this dataset and put it on a regular grid.

    EN: Two problems I found by inspecting the file rather than trusting it:
        (a) `wv (m/s)` and `max. wv (m/s)` use -9999.0 as a "sensor failed"
            marker. Left alone, a single -9999 wrecks the normalisation stats.
        (b) 327 timestamps appear twice (01-02.07.2010 and 20-21.03.2014 are
            recorded twice) and a few hundred 10-minute steps are missing
            entirely, so the series is NOT on a regular grid out of the box.
    TR: Dosyaya güvenmek yerine ona bakarak bulduğum iki sorun:
        (a) `wv (m/s)` ve `max. wv (m/s)` sütunlarında -9999.0 "sensör arızası"
            işareti var. Dokunmazsam tek bir -9999 normalizasyon
            istatistiklerini mahvediyor.
        (b) 327 zaman damgası iki kez geçiyor (01-02.07.2010 ve 20-21.03.2014
            günleri çift kaydedilmiş) ve birkaç yüz 10-dakikalık adım hiç yok;
            yani seri kutudan çıktığı haliyle düzenli bir ızgarada DEĞİL.
    """
    report = CleaningReport(rows_in=len(df))
    df = df.copy()

    # (a) sentinels -> NaN
    for col in config.SENTINEL_COLS:
        if col not in df.columns:
            continue
        mask = df[col] <= config.SENTINEL_VALUE
        n = int(mask.sum())
        if n:
            report.sentinel_cells[col] = n
            df.loc[mask, col] = np.nan

    # (b1) duplicated timestamps -> keep the first occurrence
    dup_mask = df.index.duplicated(keep="first")
    report.duplicate_timestamps = int(dup_mask.sum())
    df = df[~dup_mask]

    # (b2) reindex onto a complete grid so that gaps become explicit NaNs
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=f"{config.RAW_STEP_MINUTES}min",
    )
    before = len(df)
    df = df.reindex(full_index)
    report.missing_steps_filled = len(df) - before
    report.longest_gap_steps = _longest_nan_run(df[config.TARGET_COL])

    # EN: Remember which rows never existed BEFORE I invent values for them. The
    #     biggest gap is 74 hours in October 2016, inside the test slice, so
    #     these flags are what stop me scoring models on fabricated targets.
    # TR: Hangi satırların hiç var olmadığını, onlara değer uydurmadan ÖNCE
    #     hatırlıyorum. En büyük boşluk Ekim 2016'da 74 saat ve test diliminin
    #     içinde; dolayısıyla modelleri uydurulmuş hedefler üzerinden
    #     puanlamamı engelleyen şey tam olarak bu işaretler.
    synthetic = df[config.TARGET_COL].isna().to_numpy()

    df = df.interpolate(method="time").bfill().ffill()
    df[config.SYNTHETIC_COL] = synthetic

    report.rows_out = len(df)
    df.index.name = config.TIME_COL
    return df, report


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Down-sample 10-minute data to hourly by taking the readings on the hour.

    EN: The raw file starts at 00:10, so a naive df[::6] would give me :10
        stamps. Selecting minute == 0 gives clean hourly timestamps instead. I
        down-sample rather than average, because the target is an instantaneous
        reading and not an hourly mean.
    TR: Ham dosya 00:10'da başlıyor; dolayısıyla düz bir df[::6] bana :10'lu
        damgalar verirdi. minute == 0 seçmek temiz saatlik damgalar veriyor.
        Ortalama almak yerine seyreltiyorum, çünkü hedef saatlik ortalama değil
        anlık bir ölçüm.
    """
    return df[df.index.minute == 0].copy()


# --------------------------------------------------------------------------- #
# 3. Feature engineering / Öznitelik mühendisliği (optional)
# --------------------------------------------------------------------------- #
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn direction and clock time into features a network can actually use.

    EN: Two fixes that cost nothing and usually help:
        - Wind direction in degrees is circular: 359 and 1 are neighbours, but
          as raw numbers they sit 358 apart. I convert (wv, wd) into the wind
          vector (Wx, Wy) instead.
        - Time of day and time of year are periodic too, so I encode them as
          sin/cos pairs rather than as a raw hour counter.
    TR: Bedavaya gelen ve genelde işe yarayan iki düzeltme:
        - Derece cinsinden rüzgâr yönü döngüseldir: 359 ile 1 komşudur ama ham
          sayı olarak aralarında 358 fark var. Bunun yerine (wv, wd) ikilisini
          rüzgâr vektörüne (Wx, Wy) çeviriyorum.
        - Günün saati ve yılın zamanı da periyodiktir; bu yüzden ham saat sayacı
          yerine sin/cos çifti olarak kodluyorum.
    """
    out = df.copy()

    if "wd (deg)" in out.columns and "wv (m/s)" in out.columns:
        rad = np.deg2rad(out.pop("wd (deg)").to_numpy())
        wv = out.pop("wv (m/s)").to_numpy()
        out["Wx"] = wv * np.cos(rad)
        out["Wy"] = wv * np.sin(rad)
        if "max. wv (m/s)" in out.columns:
            max_wv = out.pop("max. wv (m/s)").to_numpy()
            out["max Wx"] = max_wv * np.cos(rad)
            out["max Wy"] = max_wv * np.sin(rad)

    seconds = np.asarray([ts.timestamp() for ts in out.index], dtype=np.float64)
    day = 24 * 60 * 60
    year = 365.2425 * day
    out["day sin"] = np.sin(seconds * (2 * np.pi / day))
    out["day cos"] = np.cos(seconds * (2 * np.pi / day))
    out["year sin"] = np.sin(seconds * (2 * np.pi / year))
    out["year cos"] = np.cos(seconds * (2 * np.pi / year))
    return out


# --------------------------------------------------------------------------- #
# 4. Chronological split / Kronolojik bölme
# --------------------------------------------------------------------------- #
def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = config.TRAIN_FRACTION,
    val_fraction: float = config.VAL_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cut the series into three contiguous blocks, oldest first.

    EN: A random split would let the model see 2016 while being scored on 2013,
        which is both leakage and a task that does not exist in reality. Train
        is the oldest block, test is the newest.
    TR: Rastgele bölme, modelin 2016'yı görüp 2013 üzerinden puanlanmasına yol
        açardı; bu hem sızıntı hem de gerçekte var olmayan bir görev. Train en
        eski blok, test en yeni blok.
    """
    n = len(df)
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return train, val, test


class Normalizer:
    """Standardise features with TRAIN statistics only.

    EN: This is the single most common leak in time-series tutorials: computing
        mean/std over the whole frame lets test-set information reach training.
        I fit on train and only transform val/test.
    TR: Zaman serisi eğitimlerindeki en yaygın sızıntı tam olarak budur:
        mean/std'yi tüm veri üzerinden hesaplamak test bilgisini eğitime taşır.
        Ben train üzerinde fit ediyorum, val/test'i yalnızca transform ediyorum.
    """

    def __init__(self, mean: pd.Series, std: pd.Series) -> None:
        self.mean = mean
        # EN: guard against a constant column producing a divide-by-zero.
        # TR: sabit bir sütunun sıfıra bölmeye yol açmasına karşı koruma.
        self.std = std.replace(0.0, 1.0)

    @classmethod
    def fit(cls, train_df: pd.DataFrame) -> "Normalizer":
        return cls(train_df.mean(), train_df.std())

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df - self.mean) / self.std

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        mu = float(self.mean[config.TARGET_COL])
        sigma = float(self.std[config.TARGET_COL])
        return values * sigma + mu

    @property
    def target_mean(self) -> float:
        return float(self.mean[config.TARGET_COL])

    @property
    def target_std(self) -> float:
        return float(self.std[config.TARGET_COL])


# --------------------------------------------------------------------------- #
# 5. Windowing / Pencereleme
# --------------------------------------------------------------------------- #
class WindowDataset(Dataset):
    """Sliding windows over one contiguous slice of the series.

    EN: Sample i is the feature window X[i : i+L] and the target is the
        temperature at index i+L-1+H, i.e. H hours after the LAST observed
        timestep of the window. Because I build one of these per split, no
        window can ever straddle a split boundary.
    TR: i numaralı örnek, X[i : i+L] öznitelik penceresi; hedef ise i+L-1+H
        indeksindeki sıcaklık, yani pencerenin SON gözlem adımından H saat
        sonrası. Her bölme için ayrı bir tane kurduğumdan hiçbir pencere bölme
        sınırını aşamıyor.
    """

    def __init__(
        self,
        features: np.ndarray,
        target_scaled: np.ndarray,
        target_celsius: np.ndarray,
        timestamps: pd.DatetimeIndex,
        lookback: int = config.LOOKBACK_HOURS,
        horizon: int = config.HORIZON_HOURS,
        synthetic: np.ndarray | None = None,
        exclude_interpolated: bool = config.EXCLUDE_INTERPOLATED,
    ) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.target_scaled = np.asarray(target_scaled, dtype=np.float32)
        self.target_celsius = np.asarray(target_celsius, dtype=np.float32)
        self.timestamps = timestamps
        self.lookback = int(lookback)
        self.horizon = int(horizon)

        n = len(self.features)
        n_candidates = n - self.lookback - self.horizon + 1
        if n_candidates <= 0:
            raise ValueError(
                f"Slice of {n} rows is too short for lookback={lookback} and "
                f"horizon={horizon}; need at least {lookback + horizon} rows."
            )

        self.sample_index = self._select_samples(
            n_candidates, synthetic, exclude_interpolated
        )
        self.n_dropped = n_candidates - len(self.sample_index)
        if len(self.sample_index) == 0:
            raise ValueError(
                "Every candidate window touches interpolated data; nothing left "
                "to train on."
            )

    def _select_samples(
        self,
        n_candidates: int,
        synthetic: np.ndarray | None,
        exclude_interpolated: bool,
    ) -> np.ndarray:
        """Keep only windows that read and predict genuine observations.

        EN: A window is rejected if ANY of its `lookback` input hours was
            invented by interpolation, or if its target hour was. I use a prefix
            sum so the check stays O(n) rather than O(n * lookback).
        TR: Bir pencere, `lookback` girdi saatlerinden HERHANGİ biri
            interpolasyonla uydurulmuşsa ya da hedef saati uydurulmuşsa
            eleniyor. Kontrol O(n * lookback) değil O(n) kalsın diye ön-toplam
            kullanıyorum.
        """
        candidates = np.arange(n_candidates)
        if synthetic is None or not exclude_interpolated:
            return candidates

        synthetic = np.asarray(synthetic, dtype=bool)
        prefix = np.concatenate([[0], np.cumsum(synthetic)])

        # EN: any synthetic hour inside the input window [i, i+lookback)
        # TR: [i, i+lookback) girdi penceresi içinde herhangi bir sentetik saat
        window_has_synthetic = (
            prefix[candidates + self.lookback] - prefix[candidates]
        ) > 0
        target_is_synthetic = synthetic[candidates + self.lookback - 1 + self.horizon]

        return candidates[~(window_has_synthetic | target_is_synthetic)]

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.sample_index[i])
        x = self.features[start : start + self.lookback]
        y = self.target_scaled[start + self.lookback - 1 + self.horizon]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

    @property
    def n_features(self) -> int:
        return self.features.shape[1]

    @property
    def n_samples(self) -> int:
        return len(self.sample_index)

    @property
    def _last_observed_positions(self) -> np.ndarray:
        return self.sample_index + self.lookback - 1

    @property
    def _target_positions(self) -> np.ndarray:
        return self.sample_index + self.lookback - 1 + self.horizon

    @property
    def last_observed_celsius(self) -> np.ndarray:
        """Temperature at the final timestep of each window, in degrees C.

        EN: This is exactly what the persistence baseline predicts.
        TR: Sağduyu (persistence) baseline'ının tahmin ettiği şey tam olarak bu.
        """
        return self.target_celsius[self._last_observed_positions]

    @property
    def targets_celsius(self) -> np.ndarray:
        """Ground-truth temperature for each sample, in degrees C."""
        return self.target_celsius[self._target_positions]

    @property
    def target_times(self) -> pd.DatetimeIndex:
        return self.timestamps[self._target_positions]


# --------------------------------------------------------------------------- #
# 6. End-to-end bundle / Uçtan uca paket
# --------------------------------------------------------------------------- #
@dataclass
class DataBundle:
    train: WindowDataset
    val: WindowDataset
    test: WindowDataset
    normalizer: Normalizer
    feature_names: list[str]
    cleaning_report: CleaningReport
    hourly: pd.DataFrame

    @property
    def n_features(self) -> int:
        return self.train.n_features

    def summary(self) -> str:
        rows = [
            f"features ({len(self.feature_names)}): {', '.join(self.feature_names)}",
            f"lookback / horizon: {self.train.lookback} h / {self.train.horizon} h",
        ]
        for name in ("train", "val", "test"):
            ds: WindowDataset = getattr(self, name)
            rows.append(
                f"{name:<5} windows={len(ds):>7,}  "
                f"(dropped {ds.n_dropped:>4,} touching interpolated data)  "
                f"{ds.timestamps[0].date()} -> {ds.timestamps[-1].date()}"
            )
        return "\n".join(rows)


def build_datasets(
    use_engineered_features: bool = False,
    nrows: int | None = None,
    lookback: int = config.LOOKBACK_HOURS,
    horizon: int = config.HORIZON_HOURS,
    feature_subset: list[str] | None = None,
    exclude_interpolated: bool = config.EXCLUDE_INTERPOLATED,
) -> DataBundle:
    """Run the whole pipeline: load -> clean -> hourly -> split -> scale -> window.

    EN: `nrows` limits the raw CSV read and is what the smoke test uses to prove
        the pipeline end to end in seconds. `feature_subset` lets me run the
        univariate control experiment (temperature only) with the same code.
    TR: `nrows` ham CSV okumasını sınırlıyor; duman testi hattın uçtan uca
        çalıştığını saniyeler içinde kanıtlamak için bunu kullanıyor.
        `feature_subset` ise aynı kodla tek değişkenli (sadece sıcaklık) kontrol
        deneyini koşmamı sağlıyor.
    """
    raw = load_raw(nrows=nrows)
    cleaned, report = clean(raw)
    hourly = to_hourly(cleaned)

    # EN: pull the synthetic flags out before any feature work, so they never
    #     become a model input and never show up in a correlation matrix.
    # TR: sentetik işaretleri öznitelik işlemlerinden önce çıkarıyorum; asla
    #     model girdisi olmasınlar ve korelasyon matrisinde görünmesinler diye.
    synthetic = hourly.pop(config.SYNTHETIC_COL).astype(bool)

    if use_engineered_features:
        hourly = add_engineered_features(hourly)

    if feature_subset is not None:
        missing = [c for c in feature_subset if c not in hourly.columns]
        if missing:
            raise KeyError(f"Requested features not present: {missing}")
        if config.TARGET_COL not in feature_subset:
            raise KeyError(
                f"{config.TARGET_COL!r} must stay in the frame; it is the target."
            )
        hourly = hourly[feature_subset]

    train_df, val_df, test_df = chronological_split(hourly)
    normalizer = Normalizer.fit(train_df)

    datasets = {}
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        scaled = normalizer.transform(part)
        datasets[name] = WindowDataset(
            features=scaled.to_numpy(dtype=np.float32),
            target_scaled=scaled[config.TARGET_COL].to_numpy(dtype=np.float32),
            target_celsius=part[config.TARGET_COL].to_numpy(dtype=np.float32),
            timestamps=part.index,
            lookback=lookback,
            horizon=horizon,
            synthetic=synthetic.loc[part.index].to_numpy(),
            exclude_interpolated=exclude_interpolated,
        )

    return DataBundle(
        train=datasets["train"],
        val=datasets["val"],
        test=datasets["test"],
        normalizer=normalizer,
        feature_names=list(hourly.columns),
        cleaning_report=report,
        hourly=hourly,
    )
