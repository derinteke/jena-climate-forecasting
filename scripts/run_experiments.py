"""Command-line runner: baselines first, then every model, then one table.

EN: The notebook tells the story; this script is what I actually run when I want
    reproducible numbers without opening Jupyter. `--smoke` proves the whole
    pipeline works end to end in under a minute before I commit to real training.
TR: Hikâyeyi notebook anlatıyor; Jupyter açmadan tekrarlanabilir sayılar
    istediğimde gerçekten koşturduğum şey ise bu script. `--smoke`, gerçek
    eğitime girişmeden önce bütün hattın uçtan uca çalıştığını bir dakikadan kısa
    sürede kanıtlıyor.

Usage / Kullanım:
    python scripts/run_experiments.py --smoke
    python scripts/run_experiments.py --models LSTM GRU CNN1D Transformer
    python scripts/run_experiments.py --engineered
    python scripts/run_experiments.py --univariate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# EN: Windows consoles default to a legacy code page that mangles Turkish
#     characters. Forcing UTF-8 keeps the bilingual output readable.
# TR: Windows konsolları Türkçe karakterleri bozan eski bir kod sayfasına
#     düşüyor. UTF-8'i zorlamak iki dilli çıktıyı okunur tutuyor.
# EN: line_buffering keeps progress visible when the output is piped to a file.
# TR: line_buffering, çıktı bir dosyaya yönlendirildiğinde ilerlemeyi görünür tutuyor.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from src import config  # noqa: E402
from src.baselines import all_baselines  # noqa: E402
from src.data import build_datasets  # noqa: E402
from src.evaluate import (  # noqa: E402
    evaluate_model,
    plot_history,
    plot_model_comparison,
    plot_predictions,
    results_table,
)
from src.models import MODEL_REGISTRY, build_model, count_parameters  # noqa: E402
from src.train import get_device, set_seed, train_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        nargs="+",
        default=["Linear", "LSTM", "GRU", "CNN1D", "Transformer"],
        choices=list(MODEL_REGISTRY),
        help="Which architectures to train.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny end-to-end run: few rows, few epochs. Proves the code works.",
    )
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE)
    p.add_argument("--lookback", type=int, default=config.LOOKBACK_HOURS)
    p.add_argument("--horizon", type=int, default=config.HORIZON_HOURS)
    p.add_argument(
        "--engineered",
        action="store_true",
        help="Add wind-vector and cyclical time features.",
    )
    p.add_argument(
        "--univariate",
        action="store_true",
        help="Control experiment: temperature as the only input feature.",
    )
    p.add_argument("--tag", default="", help="Suffix for the saved figure names.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(config.SEED)

    nrows = config.SMOKE_ROWS * config.STEPS_PER_HOUR if args.smoke else None
    epochs = config.SMOKE_EPOCHS if args.smoke else args.epochs
    lookback = 24 if args.smoke else args.lookback
    horizon = 6 if args.smoke else args.horizon
    tag = args.tag or ("smoke" if args.smoke else "full")

    print("=" * 70)
    print("DEVICE / CİHAZ")
    print("=" * 70)
    device = get_device(verbose=True)

    print()
    print("=" * 70)
    print("DATA / VERİ")
    print("=" * 70)
    t0 = time.time()
    bundle = build_datasets(
        use_engineered_features=args.engineered,
        nrows=nrows,
        lookback=lookback,
        horizon=horizon,
        feature_subset=[config.TARGET_COL] if args.univariate else None,
    )
    print(bundle.cleaning_report.summary())
    print()
    print(bundle.summary())
    print(f"(built in {time.time() - t0:.1f}s)")

    print()
    print("=" * 70)
    print("BASELINES / SAĞDUYU REFERANSLARI  (computed before any training)")
    print("=" * 70)
    results, test_predictions = all_baselines(bundle)
    for name, splits in results.items():
        print(f"{name:<14} val {splits['val']} | test {splits['test']}")
    baseline_mae = results["Persistence"]["test"].mae
    print(f"\n-> The number every model must beat: {baseline_mae:.3f} C (test MAE)")

    print()
    print("=" * 70)
    print("MODELS / MODELLER")
    print("=" * 70)
    histories = {}
    for name in args.models:
        print(f"\n[{name}]")
        set_seed(config.SEED)
        model = build_model(name, n_features=bundle.n_features, lookback=lookback)
        print(f"  parameters: {count_parameters(model):,}")
        model, history = train_model(
            model,
            bundle,
            device,
            epochs=epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            verbose=True,
        )
        histories[name] = history

        val_metrics, _ = evaluate_model(model, bundle.val, bundle.normalizer, device)
        test_metrics, test_pred = evaluate_model(
            model, bundle.test, bundle.normalizer, device
        )
        results[name] = {"val": val_metrics, "test": test_metrics}
        test_predictions[name] = test_pred
        print(f"  -> val  {val_metrics}")
        print(f"  -> test {test_metrics}")

        plot_history(
            history.as_dict(), title=f"{name} ({tag})", filename=f"history_{name}_{tag}.png"
        )

    print()
    print("=" * 70)
    print("RESULTS / SONUÇLAR")
    print("=" * 70)
    table = results_table(results)
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))

    winners = table[table.get("beats_baseline", False) == True]  # noqa: E712
    print()
    if len(winners):
        best = winners.index[0]
        gain = -table.loc[best, "vs_baseline_%"]
        print(f"Best model: {best} - beats persistence by {gain:.1f}% test MAE.")
    else:
        print(
            "No model beat the persistence baseline. That is a real result, not a "
            "bug: report it as such."
        )

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.REPORTS_DIR / f"results_{tag}.csv")
    with open(config.REPORTS_DIR / f"history_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({k: v.as_dict() for k, v in histories.items()}, f, indent=2)

    plot_model_comparison(table, filename=f"model_comparison_{tag}.png")
    plot_predictions(
        bundle.test.target_times,
        bundle.test.targets_celsius,
        test_predictions,
        n_points=min(500, len(bundle.test)),
        title=f"Test set: predicted vs actual ({tag})",
        filename=f"predictions_{tag}.png",
    )
    print(f"\nFigures -> {config.FIGURES_DIR}")
    print(f"Table   -> {config.REPORTS_DIR / f'results_{tag}.csv'}")


if __name__ == "__main__":
    main()
