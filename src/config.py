"""Konfigurasyon: yollar, seriler, sehirler, model hiperparametreleri."""
from __future__ import annotations
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DATASET_DIR = os.path.join(DATA, "dataset")
WEATHER_DIR = os.path.join(DATA, "weather")
RESULTS_DIR = os.path.join(DATA, "results")

CONSUMPTION_PARQUET = os.path.join(DATASET_DIR, "tr_consumption_hourly.parquet")
WEATHER_NATIONAL_PARQUET = os.path.join(WEATHER_DIR, "histfc_national.parquet")
WEATHER_CITIES_PARQUET = os.path.join(WEATHER_DIR, "histfc_cities.parquet")

FORECAST_MASTER = os.path.join(RESULTS_DIR, "forecast_results.csv")      # her gunun tahmini
MAPE_HISTORY = os.path.join(RESULTS_DIR, "mape_history.csv")             # gunluk skor logu
BACKTEST_CSV = os.path.join(RESULTS_DIR, "backtest_sim_2025_2026.csv")   # simule backtest

TZ = "Europe/Istanbul"          # UTC+3 sabit
DECISION_HOUR_TR = "03:00"      # gunluk calisma saati (TR)

# --- Seriler ---
EPIAS_SERIES = {
    "rt-cons": {"col": "rt_cons", "vcol": "consumption", "time": True},   # gerceklesen tüketim (hedef)
    "load-plan": {"col": "load_plan", "vcol": "lep", "time": True},       # TEIAS gun oncesi (benchmark)
}

# --- OpenMeteo (ECMWF IFS) hava ---
OM_HISTO_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OM_FC_URL = "https://api.open-meteo.com/v1/forecast"
OM_MODEL = "ecmwf_ifs"
OM_VARS = ["temperature_2m", "apparent_temperature", "relative_humidity_2m",
           "shortwave_radiation", "wind_speed_10m", "precipitation", "cloud_cover"]
# nufus agirlikli temsili sehirler (2023 TUIK'e yakin)
CITIES = [
    dict(city="istanbul",   lat=41.01, lon=28.98, pop=15.9),
    dict(city="ankara",     lat=39.93, lon=32.86, pop=5.8),
    dict(city="izmir",      lat=38.42, lon=27.14, pop=4.5),
    dict(city="bursa",      lat=40.19, lon=29.06, pop=3.2),
    dict(city="antalya",    lat=36.88, lon=30.70, pop=2.7),
    dict(city="adana",      lat=37.00, lon=35.32, pop=2.3),
    dict(city="gaziantep",  lat=37.07, lon=37.38, pop=2.2),
    dict(city="konya",      lat=37.87, lon=32.48, pop=2.3),
    dict(city="samsun",     lat=41.29, lon=36.33, pop=1.4),
    dict(city="erzurum",    lat=39.90, lon=41.27, pop=0.7),
]

# --- Model ---
TRAIN_DAYS = 1000          # ~2.7 yil son pencere (5y daha iyi degildi -> kanitli)
USE_CONT = False           # uretim = BASE (v4.3); manyak CONT deneysel (regulasyon ister)
RETRAIN_EVERY = 7          # gunluk job haftalik retrain yapar? -> gercek zamanli her gun 1/7 sıklık olur
LGB_PARAMS = dict(
    n_estimators=1500, learning_rate=0.05, num_leaves=95,
    colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
    min_child_samples=50, random_state=42, verbose=-1,
)
# v4.3 + v5.2 ensemble icin her ufka iki model (temel + manyak-surekli)
USE_ENSEMBLE = True

# --- Multi-model ensemble (LGBM + XGB + CatBoost) ---
ENSEMBLE_MODELS = ["lgbm"]  # production single; per-model tuning workers override
# --- LEP (TEIAS gun-oncesi plan) ozelligi: is 17:00 TR'de kosar, LEP(T)/LEP(T+1) yayimlidir.
# H1'e scale-free oran girer: lep(gun(t),saat)/samehr_7d_24(t-48s) -> A/B: H1 %1.98->%1.89 (81g)
USE_LEP_FEATURE = True
XGB_PARAMS = dict(n_estimators=1500, learning_rate=0.05, max_depth=8, colsample_bytree=0.8, subsample=0.85, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0, gamma=0.0, tree_method="hist", random_state=42, verbosity=0)
CAT_PARAMS = dict(iterations=1500, learning_rate=0.05, depth=8, l2_leaf_reg=3.0, random_strength=1.0, bagging_temperature=0.5, border_count=128, random_seed=42, verbose=False, loss_function="RMSE")
STACKING_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
HOLIDAY_TAIL_DAYS = 2
