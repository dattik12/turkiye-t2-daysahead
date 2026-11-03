# Sistem Durumu — 2026-11-03 gecesi

Üç eksen: **yük**, **rüzgar**, **güneş**. Amaç: GÖP fiyat tahminine giren
feature'ların en iyi halde olması.

---

## 1. Yük (talep)

**Ürün:** her gün 09:00 TR, yarının 24 saati (`data/forecast/morning/latest_yarin.parquet`)

| Bileşen | Durum |
|---|---|
| Model | 3 uzman topluluk: LightGBM %15, CatBoost %25, sinir ağı %60 |
| Feature | 83 = 52 temel + 31 saatlik takvim |
| Backtest | 720 gün walk-forward: **%1.6308** |
| TEİAŞ | aynı pencerede %2.8581 |
| Kazanma | 719 günün 570'i (%79) |
| Canlı beklenti | ~%1.80 (hava lead düzeltmesiyle, ölçüldü) |

Sinir ağını çalıştıran şey: profil taban (hafta günü × saat, son 12 hafta) +
residual head + hour/dow/month embedding. Düz MLP %3.63 idi, bu yapıyla %1.71.

**Saatlik takvim** (Excel `Event_Windows`, 2.288 olay → 31 bayrak):
720 günde tümü −0.064 pp, **tatilde −0.690 pp** (51 günün 36'sı).

**Günlük hat** (`daily_run`, PTF'in `consumption_pred_mw` kolonu):
aynı takvim eklendi (`USE_HOURLY_CAL=True`). Model ailesi hâlâ LightGBM —
CI'da catboost/torch yok.

---

## 2. Rüzgar

**Ürün:** `ptf_features` içinde `wind_pred_mw`

| | WMAPE | Not |
|---|---|---|
| Spatial model (canlı) | **%12.83** | 39 il ayrı feature |
| RİTM | %13.10 | aynı pencere |
| Eski 7-il ağırlıklı | %14.50 | kaybediyordu |

60 gün eğitim / 7 gün adım / 90 gün, 2026-08-04→11-01.

**Değişen:** `WIND_FARMS.csv` 7 il / 3 bölge (etkin 3 il) idi. Artık
`WIND_FARMS_SPATIAL.csv` ile **39 il** çekiliyor (Marmara 11, Ege 5, Akdeniz 5,
İç Anadolu 7, Karadeniz 6, Doğu/GD Anadolu 5) ve her il ayrı feature.

`blend_w = 1.0` — model artık RİTM'e yaslanmıyor, tamamen kendi tahminini veriyor.

**Geri alma:** `models/wind_lgbm_d*.txt.pre_spatial` yedekleri duruyor.

**Ölçülen gerçekleşme karşılaşması** (90 gün, EPİAŞ `rt-gen-bulk`):
WMAPE %12.98, bias +%1, korelasyon 0.9587 → RİTM %13.71.

---

## 3. Güneş

**Ürün:** `ptf_features` içinde `solar_pred_mw`

**Ölçüm sorunu çözüldü, seviye sorunu çözülmedi.**

EPİAŞ `rt-gen-bulk` kaynak kırılımı veriyor (`sun`, `wind`, `total`, ...).
Daha önce "ulusal GES gerçekleşmesi yok" demiştim — **yanlıştı**.

**Kapsam:** yalnızca kayıtlı/lisanslı santraller. GES tepesi 4.2 GW (Eylül),
Türkiye kurulu gücü ~20 GW → gerçekleşme ulusal toplamın ~%2-3'ü. Rüzgar
kapsamı makul (%11.5), GES değil.

**Şekil karşılaşması (90 gün):**

| Seri | Merkez | Kayma | nRMSE | Korel. |
|---|---|---|---|---|
| Gerçekleşen (lisanslı) | 11.98 | — | — | — |
| Nüfus ağırlıklı (mevcut) | 13.31 | +1.33 | 0.1973 | 0.883 |
| Coğrafi ortalama (39 il) | 13.18 | +1.20 | 0.1825 | 0.901 |
| Güneş bölgesi (18 il) | 13.04 | +1.05 | 0.1680 | 0.918 |

**Karar: üretim değiştirilmedi.** Sebep: gerçekleşme coğrafi olarak doğu/lisanslı
ağırlıklı, ulusal filo ise lisanssız çatı (nüfusla batıda). Ölçüm hatasını
düzeltmek için gerçek üretimi bozmak yanlış olur. Ayrıca ışınım ağırlığını
değiştirmek seviyeyi de kaydırır; seviye kalibrasyonu için ulusal GES gerçeği yok.

**Bu yüzden güneşte "TEİAŞ'ı geçtik" iddiası kurulamaz.**

---

## Açık işler

1. **Ulusal GES gerçeği** — kapasite ağırlıklı ışınım + yıllık enerji çapası
   (TEİAŞ/EPDK yıllık üretim) ile seviye kalibrasyonu. Yoksa güneş ölçülemez.
2. **Günlük hattı topluluğa çevirmek** — `requirements.txt`'e catboost eklenmeli,
   CI'da doğrulanmalı. Şu an günlük hat LightGBM-only.
3. **Canlı kanıt** — sabah topluluğu ilk kez 2026-11-04 09:00'da gerçek üretimde
   koşacak; 3-5 gün skor birikmeli. `scripts/morning_scoreboard.py` otomatik dolduruyor.
4. **Tatil açığı** — tatil MAPE 4.66 vs normal 1.49 (hâlâ 3 kat).
5. **Köprü günü** (30 Ekim tipi) — çözüm yok.

---

## Bayraklar / geri alma

| Anahtar | Varsayılan | Kapatınca |
|---|---|---|
| `USE_HOURLY_CAL` | True | saatlik takvim devre dışı |
| `WIND_SPATIAL` | (eğitimde env) | 7-il ağırlıklı rüzgar |
| `ENSEMBLE` | on | sabah hattı yalnız LightGBM |
| `models/wind_lgbm_d*.txt.pre_spatial` | — | spatial öncesi rüzgar modelleri |
