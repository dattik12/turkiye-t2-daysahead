# scripts/ — envanter

Bu dizinde 80+ script var; çoğu araştırma deneyi. **Üretimde yalnızca 7 tanesi çalışır.**

## ÜRETİM (run_morning.bat bunları çağırır)

| Script | Ne yapar | Sıra |
|---|---|---|
| `pull_daily.py` | Günlük NWP çekimi — rüzgar (10 il × 3 model) + lead_weather (10 il × 7 değişken). Kota dostu, il başına 1 istek. | 1 |
| `morning_run.py` | Yük tahmini (LGBM+CatBoost+MLP topluluğu), `yarin_YYYY-MM-DD.parquet` + skor defteri | 2 |
| `morning_scoreboard.py` | Gerçekleşen + TEİAŞ planı ile skorlama; güne göre tekilleştirir (09:00 tercih) | 3 |
| `wind_live_v5.py` | Rüzgar ML canlı çıktı → `data/forecast/wind_live.parquet` | 4 |
| `solar_live.py` | Güneş ML canlı çıktı → `data/forecast/solar_live.parquet` | 5 |
| `morning_ptf.py` | 48×13 PTF kontratı → `latest.parquet` | 6 |
| `archive_contract.py` | Kontrat arşivi + sidecar (idempotent SKIP) | 7 |

> **Bağımlılık:** `morning_ptf.py` `wind_live.parquet` ve `solar_live.parquet`'i okur.
> Tek başına çalıştırılırsa bayat veriyle **tutarsız kontrat** üretir. Zinciri
> `run_morning.bat` ile çalıştır.

## DOĞRULAMA

| Script | Ne yapar |
|---|---|
| `verify_contract.py` | 17 kapı: 48 satır, 13 kolon, denge, negatif yük, gece güneşi |
| `verify_all.py` | Geniş doğrulama |
| `feat_check.py` | T+1/T+2 özellik matrisi hash testi (regresyon) |

## ÖLÇÜM / TANI

| Script | Ne yapar |
|---|---|
| `load_audit2.py` | Yük out-of-time denetimi. `CUT_Y=1` → canlı-eşdeğer (y 07:00'de kesik) |
| `load_t2_retrain.py` | T+2 per-horizon anchored eğitim + ölçüm. `HZ`/`CUT_Y` env |
| `wind_diag.py` | Rüzgar walk-forward + tahmin dump |
| `wind_errstruct.py` | Rüzgar hata yapısı (saat/havza/rejim/cut-out/rampa) |
| `solar_diag.py` | Güneş saat bazlı bias + curtailment tanısı |
| `audit_experts.py`, `audit_bias.py` | Uzman/bias denetimleri |
| `nwp_error.py` | NWP hata karakteri |

## VERİ ÇEKİMİ (tek seferlik / arşiv kurma)

`pull_multinwp.py`, `pull_batch.py`, `pull_lead_weather.py`, `pull_prov_long.py`,
`pull_prov_wind2.py`, `pull_solar_rad.py`, `pull_generation.py`

## ARAŞTIRMA (kapalı veya beklemede)

`autoloop_*.py` (tatil/L1/L2/load/objective/weather/wind/tsfm), `solar_*.py` (t6, ens,
phys, ges, mnwp, spatial, source, fit, geo, cal), `wind_*.py` (100, mnwp, probe100,
segments, spatial_wf2, band_blend, t2, vs_ritm, weight_test, wf, train), `probe_*`,
`plan_blend.py`, `weight_schemes.py`, `backtest_*.py`, `eda_segments.py`.

> **KAPALI KONULAR** (kanıtla elendi, tekrar denemeyin): tatil ağırlıklandırma,
> rejim ağırlıkları, MAPE-uyumlu amaç fonksiyonu, Chronos-2/TSFM, XGBoost, MoE,
> güneşte yeni mimari. Detay: `docs/PHASE_LOG.md`.

## BİLİNEN SORUNLU

| Script | Sorun |
|---|---|
| `check_ptf_latest.py` | Lag hatası — tazelik kontrolü yanlış sonuç verebilir |
| `load_audit.py` | Canlı API kullanıyordu (geçmiş tarih için yanlış). `load_audit2.py` ile değiştirildi |
