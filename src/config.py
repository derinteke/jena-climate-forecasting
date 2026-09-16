"""Central configuration / Merkezi yapılandırma.

EN: Every path, hyper-parameter and split ratio lives here, so the notebook, the
    scripts and any future test all read exactly the same numbers. If I want to
    change the forecasting task, this is the only file I touch.
TR: Bütün yollar, hiper-parametreler ve bölme oranları burada duruyor; notebook,
    scriptler ve ileride yazacağım testler aynı sayıları okusun diye. Tahmin
    görevini değiştirmek istersem dokunmam gereken tek dosya bu.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths / Yollar
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

DATASET_URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    "jena_climate_2009_2016.csv.zip"
)
ZIP_PATH = RAW_DIR / "jena_climate_2009_2016.csv.zip"
CSV_PATH = RAW_DIR / "jena_climate_2009_2016.csv"

# --------------------------------------------------------------------------- #
# Dataset facts / Veri seti gerçekleri
# --------------------------------------------------------------------------- #
TIME_COL = "Date Time"
TIME_FORMAT = "%d.%m.%Y %H:%M:%S"
TARGET_COL = "T (degC)"

# EN: The raw file is sampled every 10 minutes -> 6 rows per hour.
# TR: Ham dosya 10 dakikada bir örneklenmiş -> saatte 6 satır.
RAW_STEP_MINUTES = 10
STEPS_PER_HOUR = 60 // RAW_STEP_MINUTES

# EN: Columns that carry a -9999.0 "sensor failed" sentinel instead of NaN.
# TR: NaN yerine -9999.0 "sensör arızalandı" değeri taşıyan sütunlar.
SENTINEL_VALUE = -9999.0
SENTINEL_COLS = ["wv (m/s)", "max. wv (m/s)"]

# --------------------------------------------------------------------------- #
# Forecasting task / Tahmin görevi
# --------------------------------------------------------------------------- #
# EN: I look at the last 120 hourly observations (5 days) and predict the single
#     temperature reading 24 hours after the end of that window.
# TR: Son 120 saatlik gözleme (5 gün) bakıp, o pencerenin bitiminden 24 saat
#     sonraki tek sıcaklık değerini tahmin ediyorum.
LOOKBACK_HOURS = 120
HORIZON_HOURS = 24

# --------------------------------------------------------------------------- #
# Splits / Bölmeler
# --------------------------------------------------------------------------- #
# EN: Chronological, never random. Train is the oldest slice, test the newest.
# TR: Kronolojik, asla rastgele değil. Train en eski dilim, test en yeni.
TRAIN_FRACTION = 0.60
VAL_FRACTION = 0.20
TEST_FRACTION = 0.20

# --------------------------------------------------------------------------- #
# Training / Eğitim
# --------------------------------------------------------------------------- #
SEED = 42
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 0  # EN: 0 is fastest on Windows. / TR: Windows'ta 0 en hızlısı.

# --------------------------------------------------------------------------- #
# Smoke test / Duman testi
# --------------------------------------------------------------------------- #
# EN: A tiny end-to-end run to prove the pipeline works before the real training.
# TR: Gerçek eğitimden önce hattın çalıştığını kanıtlayan minik uçtan uca koşu.
SMOKE_ROWS = 6000
SMOKE_EPOCHS = 2

# --------------------------------------------------------------------------- #
# Interpolated rows / İnterpole edilmiş satırlar
# --------------------------------------------------------------------------- #
# EN: The raw series has real gaps - the largest is 74 hours in October 2016,
#     which falls inside the TEST slice. Interpolating across three days of
#     weather invents data, and scoring a model on invented targets is not an
#     honest test. So I flag every synthetic row and, by default, drop any window
#     that reads one or predicts one.
# TR: Ham seride gerçek boşluklar var - en büyüğü Ekim 2016'da 74 saat ve tam
#     TEST diliminin içine düşüyor. Üç günlük havayı interpole etmek veri
#     uydurmaktır ve uydurulmuş hedefler üzerinden model puanlamak dürüst bir
#     sınav değildir. Bu yüzden her sentetik satırı işaretliyorum ve varsayılan
#     olarak böyle bir satırı okuyan ya da tahmin eden pencereleri atıyorum.
SYNTHETIC_COL = "_synthetic"
EXCLUDE_INTERPOLATED = True
