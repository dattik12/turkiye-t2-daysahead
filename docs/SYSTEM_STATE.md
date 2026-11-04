# Sistem Durumu — 2026-11-04

Üç eksen: **yük**, **rüzgar**, **güneş**. Amaç: GÖP fiyat tahminine giren
feature'ların en iyi halde olması.

---

## 1. Yük (talep)

**Ürün:** her gün 09:00 TR, yarının 24 saati (`data/forecast/morning/latest_yarin.parquet`)

| Bileşen | Durum |
|---|---|
| Model | 3 uzman topluluk: LightGBM %10, CatBoost %20, sinir ağı %70 |
| Eğitim | son **2000 gün** (pencere taraması: tepe 2000; 4500 gün felaket) |
| Sinir ağı | **3 seed** [7,17,27] ortalaması (varyans azaltma, 5 seed doyum) |
| Feature | 83 = 52 temel + 31 saatlik takvim |
| Backtest | 720 gün walk-forward: **%1.5851** |
| TEİAŞ | aynı pencerede %2.8550 |
| Canlı beklenti | **~%1.63** (hava lead cezası +0.041, ölçüldü) |

**Canlı ile backtest birebir aynı mı?** Model, feature, ağırlık, pencere ve lag kuralı
aynı; iki fark var ve ikisi de bilinçli:

| | Backtest | Canlı |
|---|---|---|
| Yeniden eğitim | 30 günde bir | **her gün** (taze veri) |
| Hava | analiz (historical-forecast) | **IFS tahmini** (lead'li) |

Hava farkının maliyeti ölçüldü: **+0.041 pp** (720 gün, 4 kollu deney).
Eğitim sıklığı farkı ölçülmedi — canlı her gün daha taze veriyle eğitiyor, backtest
30 günde bir. Yönü bilinmiyor; 720 günlük sayı 30 günlük kadans varsayımıyla geçerli.

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
3. **Canlı kanıt** — model 2026-11-03'ten beri canlıda; skor defterinde henüz
   puanlanmış gün yok (gerçekleşmeler hedef gün tamamlanınca doluyor).
4. **Eğitim sıklığı farkı** — canlı her gün, backtest 30 günde bir yeniden eğitiyor.
   Yönü ölçülmedi; 720 günlük skor 30 günlük kadansa ait.

---

## Kapanan kollar (2026-11, ~80 hipotez)

Aşağıdakiler ölçüldü ve **hiçbiri kazanmadı** — tekrar denemeye gerek yok.
Detay: `autoloop/ledger_*.jsonl`, scriptler `scripts/autoloop_*.py` + yardımcılar.

| Kol | Ne denendi | Sonuç |
|---|---|---|
| Tatil | 7 örnek ağırlık şeması (×2…×5, komşuluk, köprü, çıkarma) | Nötr/negatif; hata yapısal (yılda ~10 gün) |
| Hava | lead-1 ile eğitim, sıcaklık bias düzeltmesi | Açık sadece 0.041 pp; düzeltme zarar veriyor |
| Ağırlıklar | 6 rejim şeması (mevsim/ay/hafta sonu/tatil) | Sabit ağırlık optimal |
| TSFM | Chronos-2 (720 gün, aynı günler) | En iyi karışım ağırlığı **0** |
| Amaç fonksiyonu | L1, Huber, 1/y ve 1/√y ağırlığı | L2 optimal |
| TEİAŞ planı | 4. uzman, walk-forward karışım | −0.005…−0.007 (eşik altı, gürültü) |
| Model ailesi | XGBoost, büyük MLP, 6-uzman MoE, Ridge | Hepsi gerileme |
| Rüzgar | 13 hipotez (pencere, seed, ağaç, monotoni) | 90 gün zaten optimal |

**Özü:** kazanç model karmaşıklığından değil, **daha çok geçmiş + varyans azaltmadan**
geliyor. 2000 gün + 3-seed MLP bu turun tek gerçek kazancı (−0.043 pp, 720 gün).

---

## Bayraklar / geri alma

| Anahtar | Varsayılan | Kapatınca |
|---|---|---|
| `USE_HOURLY_CAL` | True | saatlik takvim devre dışı |
| `WIND_SPATIAL` | (eğitimde env) | 7-il ağırlıklı rüzgar |
| `ENSEMBLE` | on | sabah hattı yalnız LightGBM |
| `models/wind_lgbm_d*.txt.pre_spatial` | — | spatial öncesi rüzgar modelleri |
