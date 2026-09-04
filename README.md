# ⚡ TR Physical Balance & Merit-Order Store (`T+1` / `T+2`)

> **Operational, dual-cron physical grid balance engine delivering hourly day-ahead demand, solar, wind, residual load, and ramp dynamics for EPİAŞ power market modeling.**

[![daily-t2-forecast](https://github.com/dattik12/turkiye-t2-daysahead/actions/workflows/daily.yml/badge.svg)](https://github.com/dattik12/turkiye-t2-daysahead/actions/workflows/daily.yml)

---

## ⚡ Downstream Handoff (Agent & Model Ready)

Downstream fiyat (PTF) modelleri ve otonom ajanlar için doğrulanmış canlı veri kontratı:

```python
from src.features.ptf_store import load_ptf_features

# Returns strictly validated 48-hour block (T+1 & T+2)
df = load_ptf_features()
```

```text
Contract Invariants:
- 48 continuous hourly rows (no duplicates, no gaps, Europe/Istanbul)
- residual_load_mw = consumption_pred_mw - (solar_pred_mw + wind_pred_mw)
- Strict bounds: residual_load_mw > 0 | night_solar == 0.0 | tolerance < 0.01 MW
```

---

## 📊 Core Benchmarks

Production-grade verified metrics against official system operators:

| Component | Architecture / Method | Benchmark | Production Metric |
| --- | --- | --- | --- |
| **Demand (Load)** | v4.5 LightGBM + Gated Regime MoE (ECMWF IFS) | TEİAŞ Plan (%3.01) | **T+1 %1.38 / T+2 %2.11** *(30-day A/B)* |
| **Solar (GES)** | 27.4 GW ETKB cap, PR=0.921, Zenith night mask | 66 Licensed Plants | **r = 0.88 shape corr** *(Tem'26 anchor)* |
| **Wind (RES)** | 7-Hub NNLS spatial weights + Dual-LGBM | RİTM Forecast (%9.57) | **%9.22 MAPE** *(Spread-blend)* |
| **Long-Term** | 597 Decision Days Backtest (14,280 points) | TEİAŞ Plan (%3.01) | **T+1 %1.68 / T+2 %2.26** *(597-day)* |

---

## 📋 Data Contract (`latest.parquet`)

Her koşuda `data/forecast/ptf_features/latest.parquet` ve snapshot arşivine yazılan 13 kolon:

```text
data/forecast/ptf_features/latest.parquet
├── Temporal
│   ├── datetime                  datetime64[Europe/Istanbul] (PK)
│   └── horizon                   category ('T+1', 'T+2')
├── Physical Quantities (MW, float32)
│   ├── consumption_pred_mw       National load forecast (v4.5 MoE)
│   ├── solar_pred_mw             Calibrated PV production
│   ├── wind_pred_mw              NNLS dual-blend wind forecast
│   ├── renewable_generation_mw   solar + wind
│   └── residual_load_mw          Net system load (merit-order driver)
├── Market Dynamics & Ramps
│   ├── renewable_penetration     float32 [0.0, 1.0)
│   ├── residual_ramp_1h          float32 (1-hour net load derivative, MW/h)
│   ├── solar_ramp_1h             float32 (Sunset/sunrise ramp, MW/h)
│   └── is_peak_hour              uint8 (1 if 08:00 <= hour < 20:00 else 0)
└── System Quality Flags (category)
    ├── solar_status              'ok' | 'zero_night' | 'unconfigured'
    └── wind_status               'blend' | 'ritm' | 'fallback'
```

> ⚠️ **Fail-Fast:** kontrat ihlalinde (negatif yük, gece GES üretimi, toplamsallık sapması) sessizce bozuk veri yazılmaz; `validate()` pipeline'ı durdurur.

---

## 🔄 Pipeline Lifecycle (Dual-Cron)

```text
[09:30 TSİ] Day-Ahead Base (GÖP Öncesi)
├── Ingestion: EPİAŞ (rt-cons, load-plan, RİTM) + ECMWF IFS (00:00 UTC)
├── Modeling : Load v4.5 + Solar PR-fit + Wind NNLS blend
└── Export   : latest.parquet + ptf_features_YYYY-MM-DD.parquet

[17:00 TSİ] Post-LEP Refresh (GİP & T+2 Güncellemesi)
├── Ingestion: Latest TEİAŞ LEP schedule + ECMWF IFS (06:00 UTC)
└── Update   : Refreshed T+1 evening & sharpened T+2 horizon
```

---

## 🛠️ Quickstart

```bash
# 1. Setup environment (lightgbm 4.5.0 + numpy<2, kritik combo)
pip install -r requirements.txt

# 2. Configure credentials in .env (EPİAŞ Şeffaflık)
# EPTR_USERNAME=your_username
# EPTR_PASSWORD=your_password

# 3. Manual pipeline execution
python -m scripts.bootstrap            # ilk tam veri (2016→bugün)
python -m scripts.daily_run            # günlük tahmin (dual-cron orkestrasyonu)
python -m scripts.build_ptf_features   # 13 kolonluk store'u yeniden kur + validate
```

GitHub Actions için `EPTR_USERNAME` ve `EPTR_PASSWORD` → **Settings → Secrets**.

---

## 📂 Repository Topology

```text
├── src/
│   ├── config.py / data.py / forecast.py / models.py / scoring.py
│   ├── consumption/      # v4.3 tüketim hattı referans paketi
│   ├── features/         # Feature engineering + ptf_store validation engine
│   ├── solar/            # Physical solar model (zenith, thermal derating)
│   └── wind/             # 7-Hub spatial matrix & RİTM blend model
├── scripts/
│   ├── daily_run.py          # Dual-cron orchestrator
│   ├── build_ptf_features.py # 13-col contract builder & sanity gate
│   ├── collect_fcast_snapshot.py  # Lead-time aligned IFS archive
│   └── check_ptf_latest.py   # Sert kontrat denetimi (stale/bozukta kırmızı)
└── data/forecast/ptf_features/
    ├── latest.parquet    # Single production entrypoint for downstream
    └── archive/          # Immutable lead-time aligned snapshots
```

---

## 📈 Live Evidence & Sources

![Performance](docs/performance.png)

* **Backtest exports** (`data/exports/`): `backtest_t1_t2_2025_2026.csv` (14.280 satır: gerçekleşen + T+1 + T+2 + TEİAŞ planı) ve `backtest_daily_summary_2025_2026.csv` — yeniden üretim: `python -m scripts.export_backtest`, grafik: `python -m scripts.make_readme_chart`.
* **Canlı kanıt defteri:** her koşunun tahmini gerçekleşince `mape_history.csv`'e düşer; tahminler ve skorlar Actions tarafından otomatik commitlenir.
* **Veri kaynakları:** EPİAŞ Şeffaflık (`eptr2`) — `rt-cons` (hedef), `load-plan` (TEİAŞ, rakip), RİTM (rüzgar); OpenMeteo `ecmwf_ifs` — 10 il nüfus-ağırlıklı hava.
* **Leak disiplini:** ufuk-güvenli lag setleri (H1: lag24+, H2: lag72+), `scripts/_leak_check.py` ile statik + dinamik denetim.
