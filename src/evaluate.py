"""Metrics and plots, always reported in degrees Celsius.

EN: Everything the network sees is standardised, so a raw MSE of 0.13 means
    nothing to a human. Every number that leaves this module is converted back
    to degrees C, because "the model is off by 2.3 degrees" is a claim I can
    actually check against reality.
TR: Ağın gördüğü her şey standartlaştırılmış durumda; dolayısıyla 0.13'lük ham
    bir MSE insana hiçbir şey ifade etmiyor. Bu modülden çıkan her sayı derece
    cinsine geri çevriliyor, çünkü "model 2.3 derece şaşırıyor" cümlesi gerçekle
    kıyaslayabileceğim bir iddia.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import config


# --------------------------------------------------------------------------- #
# Metrics / Metrikler
# --------------------------------------------------------------------------- #
@dataclass
class Metrics:
    """MAE and RMSE in degrees C / Derece cinsinden MAE ve RMSE."""

    mae: float
    rmse: float
    n: int

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"MAE {self.mae:.3f} C | RMSE {self.rmse:.3f} C | n={self.n:,}"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    err = y_pred - y_true
    return Metrics(
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err**2))),
        n=int(y_true.size),
    )


# --------------------------------------------------------------------------- #
# Model predictions / Model tahminleri
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, dataset, normalizer, device, batch_size: int = 512) -> np.ndarray:
    """Run the model over a dataset and return predictions in degrees C."""
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS
    )
    preds = []
    for x, _ in loader:
        out = model(x.to(device)).squeeze(-1)
        preds.append(out.detach().cpu().numpy())
    scaled = np.concatenate(preds)
    return normalizer.inverse_transform_target(scaled)


def evaluate_model(
    model, dataset, normalizer, device, batch_size: int = 512
) -> tuple[Metrics, np.ndarray]:
    """Return (metrics in C, predictions in C) for one dataset."""
    y_pred = predict(model, dataset, normalizer, device, batch_size)
    y_true = dataset.targets_celsius
    return compute_metrics(y_true, y_pred), y_pred


# --------------------------------------------------------------------------- #
# Results table / Sonuç tablosu
# --------------------------------------------------------------------------- #
def results_table(results: dict, baseline_name: str = "Persistence") -> pd.DataFrame:
    """Turn {name: {'val': Metrics, 'test': Metrics}} into a ranked table.

    EN: The `beats_baseline` column is the whole point of this project. A model
        that does not beat persistence is a failed model, however pretty its
        loss curve looks.
    TR: `beats_baseline` sütunu bu projenin bütün meselesi. Persistence'ı
        geçemeyen model, kayıp eğrisi ne kadar güzel görünürse görünsün,
        başarısız bir modeldir.
    """
    rows = []
    for name, splits in results.items():
        row = {"model": name}
        for split_name, metrics in splits.items():
            if metrics is None:
                continue
            row[f"{split_name}_mae"] = metrics.mae
            row[f"{split_name}_rmse"] = metrics.rmse
        rows.append(row)

    table = pd.DataFrame(rows).set_index("model")

    if baseline_name in table.index and "test_mae" in table.columns:
        base_mae = table.loc[baseline_name, "test_mae"]
        table["vs_baseline_%"] = (table["test_mae"] - base_mae) / base_mae * 100.0
        table["beats_baseline"] = table["test_mae"] < base_mae

    sort_key = "test_mae" if "test_mae" in table.columns else table.columns[0]
    return table.sort_values(sort_key)


# --------------------------------------------------------------------------- #
# Plots / Grafikler
# --------------------------------------------------------------------------- #
def _save(fig, filename: str | None):
    if filename:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES_DIR / filename, dpi=140, bbox_inches="tight")
    return fig


def plot_history(history: dict, title: str = "", filename: str | None = None):
    """Training vs validation curves, with validation MAE in degrees C."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE (standardised)")
    axes[0].set_title("Loss / Kayıp")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["val_mae_celsius"], color="tab:red", label="val MAE")
    if history.get("best_epoch") is not None:
        axes[1].axvline(
            history["best_epoch"], ls="--", c="gray", label="best epoch / en iyi epoch"
        )
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MAE (degrees C)")
    axes[1].set_title("Validation MAE / Doğrulama MAE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, filename)


def plot_predictions(
    times,
    y_true: np.ndarray,
    predictions: dict,
    n_points: int = 500,
    title: str = "",
    filename: str | None = None,
):
    """Predicted vs actual temperature over a slice of the test set."""
    fig, ax = plt.subplots(figsize=(14, 5))
    sl = slice(0, n_points)

    ax.plot(times[sl], y_true[sl], label="actual / gerçek", color="black", lw=2)
    for name, y_pred in predictions.items():
        ax.plot(times[sl], np.asarray(y_pred)[sl], label=name, alpha=0.8, lw=1.2)

    ax.set_ylabel("T (degrees C)")
    ax.set_xlabel("time / zaman")
    ax.set_title(title or "Predicted vs actual / Tahmin ve gerçek")
    ax.legend(ncol=3)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, filename)


def plot_scatter(
    y_true: np.ndarray, y_pred: np.ndarray, name: str, filename: str | None = None
):
    """Scatter of predicted against actual, with the ideal y = x line."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=3, alpha=0.15)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="perfect / kusursuz")
    ax.set_xlabel("actual T (degrees C) / gerçek")
    ax.set_ylabel("predicted T (degrees C) / tahmin")
    ax.set_title(name)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, filename)


def plot_error_distribution(
    errors: dict, filename: str | None = None, bins: int = 80
):
    """Histogram of signed errors, so bias shows up as an off-centre peak."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, err in errors.items():
        ax.hist(err, bins=bins, histtype="step", lw=1.6, label=name, density=True)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("prediction error (degrees C) / tahmin hatası")
    ax.set_ylabel("density / yoğunluk")
    ax.set_title("Error distribution / Hata dağılımı")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, filename)


def plot_model_comparison(table: pd.DataFrame, filename: str | None = None):
    """Bar chart of test MAE with the baseline drawn as a red line."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ordered = table.sort_values("test_mae")
    colors = [
        "tab:red" if name == "Persistence" else "tab:blue" for name in ordered.index
    ]
    ax.bar(ordered.index, ordered["test_mae"], color=colors)
    if "Persistence" in table.index:
        base = float(table.loc["Persistence", "test_mae"])
        ax.axhline(base, ls="--", c="tab:red", label=f"baseline {base:.3f} C")
        ax.legend()
    ax.set_ylabel("test MAE (degrees C)")
    ax.set_title("Model comparison / Model karşılaştırması")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return _save(fig, filename)
