"""Common-sense baselines the deep models have to beat.

EN: This is the honesty module. On the Jena dataset the naive rule "the
    temperature in 24 hours will be the temperature right now" is famously hard
    to beat, and a neural network that cannot beat it has learned nothing
    useful. I compute the baselines FIRST, before training anything, so I am not
    tempted to grade the models on a curve afterwards.
TR: Bu, dürüstlük modülü. Jena veri setinde "24 saat sonraki sıcaklık, şu anki
    sıcaklıktır" şeklindeki naif kuralı geçmek meşhur şekilde zordur; bunu
    geçemeyen bir sinir ağı işe yarar hiçbir şey öğrenmemiştir. Baseline'ları
    hiçbir şey eğitmeden ÖNCE hesaplıyorum ki sonradan modelleri kayırma
    eğilimine kapılmayayım.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data import WindowDataset
from src.evaluate import Metrics, compute_metrics


# --------------------------------------------------------------------------- #
# 1. Persistence / Sağduyu baseline'ı
# --------------------------------------------------------------------------- #
def persistence_predictions(dataset: WindowDataset) -> np.ndarray:
    """Predict "same as the last observed value" / "son gözlemle aynı" tahmini.

    EN: The forecast for t+24h is simply the temperature at t. No parameters, no
        training, no way to overfit. This is the number to beat.
    TR: t+24s için tahmin, doğrudan t anındaki sıcaklık. Parametre yok, eğitim
        yok, overfit etme ihtimali yok. Geçilmesi gereken sayı bu.
    """
    return dataset.last_observed_celsius.copy()


def persistence_metrics(dataset: WindowDataset) -> tuple[Metrics, np.ndarray]:
    y_pred = persistence_predictions(dataset)
    return compute_metrics(dataset.targets_celsius, y_pred), y_pred


# --------------------------------------------------------------------------- #
# 2. Constant mean / Sabit ortalama
# --------------------------------------------------------------------------- #
def mean_predictions(dataset: WindowDataset, train_mean_celsius: float) -> np.ndarray:
    """Always predict the training-set mean temperature.

    EN: The weakest sensible reference. If a model cannot beat THIS, something is
        broken in the pipeline rather than in the architecture.
    TR: En zayıf makul referans. Bir model BUNU bile geçemiyorsa sorun mimaride
        değil, hattın kendisindedir.
    """
    return np.full(len(dataset), float(train_mean_celsius), dtype=np.float32)


def mean_metrics(
    dataset: WindowDataset, train_mean_celsius: float
) -> tuple[Metrics, np.ndarray]:
    y_pred = mean_predictions(dataset, train_mean_celsius)
    return compute_metrics(dataset.targets_celsius, y_pred), y_pred


# --------------------------------------------------------------------------- #
# 3. Seasonal climatology / Mevsimsel iklim ortalaması
# --------------------------------------------------------------------------- #
class ClimatologyBaseline:
    """Average temperature for each (month, hour) cell, learned from train only.

    EN: A step up from the constant mean: it knows that July is warmer than
        January and that 15:00 is warmer than 05:00, but it knows nothing about
        current weather. It is a fair test of how much of a model's skill is
        just "learning the seasons".
    TR: Sabit ortalamanın bir üstü: temmuzun ocaktan, saat 15:00'in 05:00'ten
        daha sıcak olduğunu biliyor ama güncel hava durumu hakkında hiçbir fikri
        yok. Bir modelin başarısının ne kadarının sadece "mevsimleri öğrenmek"
        olduğunu ölçmek için dürüst bir sınav.
    """

    def __init__(self) -> None:
        self.table: pd.Series | None = None
        self.fallback: float = 0.0
        # EN: Fraction of requested (month, hour) cells that train never saw. If
        #     this is high the baseline silently degenerates into TrainMean, so
        #     I track it rather than letting the two look mysteriously equal.
        # TR: İstenen (ay, saat) hücrelerinin train'de hiç görülmeyen oranı. Bu
        #     yüksekse baseline sessizce TrainMean'e dönüşür; ikisi gizemli
        #     şekilde eşit görünmesin diye bunu takip ediyorum.
        self.missing_fraction: float = 0.0

    def fit(self, train_hourly: pd.DataFrame) -> "ClimatologyBaseline":
        temps = train_hourly[config.TARGET_COL]
        key = [temps.index.month, temps.index.hour]
        self.table = temps.groupby(key).mean()
        self.table.index.names = ["month", "hour"]
        self.fallback = float(temps.mean())
        return self

    def predict(self, times: pd.DatetimeIndex) -> np.ndarray:
        if self.table is None:
            raise RuntimeError("Call fit() before predict().")
        keys = pd.MultiIndex.from_arrays([times.month, times.hour])
        values = self.table.reindex(keys).to_numpy(dtype=np.float64)
        self.missing_fraction = float(np.isnan(values).mean())
        return np.nan_to_num(values, nan=self.fallback).astype(np.float32)

    def metrics(self, dataset: WindowDataset) -> tuple[Metrics, np.ndarray]:
        y_pred = self.predict(dataset.target_times)
        return compute_metrics(dataset.targets_celsius, y_pred), y_pred


# --------------------------------------------------------------------------- #
# Convenience / Kolaylık
# --------------------------------------------------------------------------- #
def all_baselines(bundle) -> dict:
    """Evaluate every baseline on val and test in one call.

    EN: Returns {name: {'val': Metrics, 'test': Metrics}} plus the raw test
        predictions, so the notebook can plot them next to the models.
    TR: {isim: {'val': Metrics, 'test': Metrics}} ve ham test tahminlerini
        döndürüyor; böylece notebook bunları modellerin yanına çizebiliyor.
    """
    train_mean = float(bundle.normalizer.target_mean)

    climatology = ClimatologyBaseline().fit(
        bundle.hourly.loc[: bundle.train.timestamps[-1]]
    )

    results: dict = {}
    test_preds: dict = {}

    for name, fn in (
        ("Persistence", lambda ds: persistence_metrics(ds)),
        ("TrainMean", lambda ds: mean_metrics(ds, train_mean)),
        ("Climatology", lambda ds: climatology.metrics(ds)),
    ):
        val_metrics, _ = fn(bundle.val)
        test_metrics, test_pred = fn(bundle.test)
        results[name] = {"val": val_metrics, "test": test_metrics}
        test_preds[name] = test_pred

    if climatology.missing_fraction > 0.01:
        print(
            f"NOTE: Climatology had no train data for "
            f"{climatology.missing_fraction:.0%} of the test (month, hour) cells, "
            f"so it fell back to the train mean there. Expected on short slices "
            f"such as the smoke run; suspicious on the full dataset."
        )

    return results, test_preds
