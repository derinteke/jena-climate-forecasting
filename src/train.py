"""Training loop, device handling and early stopping.

EN: One loop trains every architecture, because all five models share the same
    (batch, lookback, n_features) -> (batch, 1) signature. The loop reports
    validation MAE in degrees Celsius every epoch, so I can see the real-world
    error while training rather than only a standardised MSE.
TR: Tek bir döngü bütün mimarileri eğitiyor; çünkü beş modelin de imzası aynı:
    (batch, lookback, n_features) -> (batch, 1). Döngü her epoch'ta doğrulama
    MAE'sini derece cinsinden raporluyor; böylece eğitim sırasında yalnızca
    standartlaştırılmış MSE'yi değil gerçek dünyadaki hatayı da görüyorum.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src import config


# --------------------------------------------------------------------------- #
# Reproducibility and device / Tekrarlanabilirlik ve cihaz
# --------------------------------------------------------------------------- #
def set_seed(seed: int = config.SEED) -> None:
    """Seed Python, NumPy and Torch / Python, NumPy ve Torch tohumlarını sabitle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(verbose: bool = True) -> torch.device:
    """Pick CUDA when available and print proof of what is actually being used.

    EN: I print the device explicitly because "I assumed it was on the GPU" is
        one of the easiest ways to waste an afternoon. On this machine the card
        is a GTX 1050 Ti (compute capability sm_61), which needs a CUDA 12.6
        build of PyTorch: the newer cu128 wheels no longer compile for Pascal.
    TR: Cihazı açıkça yazdırıyorum, çünkü "GPU'da koştuğunu varsaydım" cümlesi
        bir öğleden sonrayı çöpe atmanın en kolay yollarından biri. Bu makinedeki
        kart GTX 1050 Ti (hesaplama yeteneği sm_61) ve PyTorch'un CUDA 12.6
        derlemesini istiyor: daha yeni cu128 wheel'leri artık Pascal için
        derlenmiyor.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"torch            : {torch.__version__}")
        print(f"device           : {device}")
        if device.type == "cuda":
            major, minor = torch.cuda.get_device_capability(0)
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"gpu              : {torch.cuda.get_device_name(0)}")
            print(f"compute capability: sm_{major}{minor}")
            print(f"vram             : {total_gb:.1f} GB")
            print(f"built for archs  : {torch.cuda.get_arch_list()}")
        else:
            print("WARNING: running on CPU - training will be noticeably slower.")
    return device


def make_loaders(
    bundle, batch_size: int = config.BATCH_SIZE
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build the three loaders. Only train is shuffled.

    EN: Shuffling windows inside the training slice is fine and helps SGD; the
        chronological guarantee lives in the split itself, not in the sampler.
        Validation and test stay in order so the prediction plots read as time.
    TR: Eğitim dilimi İÇİNDE pencereleri karıştırmak sorunsuz ve SGD'ye yarıyor;
        kronolojik garanti örnekleyicide değil, bölmenin kendisinde duruyor.
        Doğrulama ve test sırada kalıyor ki tahmin grafikleri zaman gibi okunsun.
    """
    common = dict(num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(bundle.train, batch_size=batch_size, shuffle=True, **common),
        DataLoader(bundle.val, batch_size=batch_size, shuffle=False, **common),
        DataLoader(bundle.test, batch_size=batch_size, shuffle=False, **common),
    )


# --------------------------------------------------------------------------- #
# Training / Eğitim
# --------------------------------------------------------------------------- #
@dataclass
class History:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_mae_celsius: list = field(default_factory=list)
    epoch_seconds: list = field(default_factory=list)
    best_epoch: int | None = None
    best_val_mae: float = float("inf")
    stopped_early: bool = False

    def as_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_mae_celsius": self.val_mae_celsius,
            "epoch_seconds": self.epoch_seconds,
            "best_epoch": self.best_epoch,
            "best_val_mae": self.best_val_mae,
            "stopped_early": self.stopped_early,
        }


@torch.no_grad()
def _validate(model, loader, criterion, device, target_std: float):
    model.eval()
    total_loss = 0.0
    total_abs = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x).squeeze(-1)
        total_loss += criterion(out, y).item() * y.size(0)
        total_abs += (out - y).abs().sum().item()
        n += y.size(0)
    # EN: MAE is linear, so scaling the standardised MAE by the target std gives
    #     the MAE in degrees C exactly; the means cancel out.
    # TR: MAE doğrusal olduğundan, standartlaştırılmış MAE'yi hedefin std'siyle
    #     çarpmak tam olarak derece cinsinden MAE'yi verir; ortalamalar sadeleşir.
    return total_loss / n, (total_abs / n) * target_std


def train_model(
    model: nn.Module,
    bundle,
    device: torch.device,
    epochs: int = config.EPOCHS,
    lr: float = config.LEARNING_RATE,
    batch_size: int = config.BATCH_SIZE,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    verbose: bool = True,
) -> tuple[nn.Module, History]:
    """Train one model and restore the weights from its best validation epoch.

    EN: Model selection happens on validation only; the test set is not looked
        at until the very end, once per model. Early stopping restores the best
        state so the reported test score belongs to the checkpoint I would
        actually ship.
    TR: Model seçimi yalnızca doğrulama üzerinden yapılıyor; test setine en sonda,
        model başına bir kez bakılıyor. Erken durdurma en iyi durumu geri yüklüyor
        ki raporlanan test skoru, gerçekten kullanacağım kontrol noktasına ait
        olsun.
    """
    train_loader, val_loader, _ = make_loaders(bundle, batch_size)
    target_std = bundle.normalizer.target_std

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, patience // 2)
    )

    history = History()
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x).squeeze(-1), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * y.size(0)
            n += y.size(0)

        train_loss = running / n
        val_loss, val_mae = _validate(model, val_loader, criterion, device, target_std)
        scheduler.step(val_loss)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_mae_celsius.append(val_mae)
        history.epoch_seconds.append(time.time() - t0)

        improved = val_mae < history.best_val_mae - 1e-4
        if improved:
            history.best_val_mae = val_mae
            history.best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            flag = " *" if improved else ""
            print(
                f"  epoch {epoch + 1:>3}/{epochs}  "
                f"train {train_loss:.4f}  val {val_loss:.4f}  "
                f"val MAE {val_mae:.3f} C  "
                f"({history.epoch_seconds[-1]:.1f}s){flag}"
            )

        if epochs_without_improvement >= patience:
            history.stopped_early = True
            if verbose:
                print(
                    f"  early stop at epoch {epoch + 1} "
                    f"(best epoch {history.best_epoch + 1}, "
                    f"{history.best_val_mae:.3f} C)"
                )
            break

    model.load_state_dict(best_state)
    return model.to(device), history
