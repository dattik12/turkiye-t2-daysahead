# PHASE 1 — Honest baseline + ölçüm

**Lab modu.** Prod kod/config/schedule değiştirilmedi, hiçbir şey deploy edilmedi.
Yeni veri yeni path'e yazılıyor (`data/weather/honest/`). Mevcut arşivlerin üzerine yazılmadı.
**Model tuning / feature denemesi yapılmadı** — yalnız hava kaynağı değiştirilerek dürüst taban ölçülüyor.

Tarih: 2026-11-15 · Script'ler: `scripts/phase1_*.py`

---

## P1.0 — SANITY

### P1.0a · `TRAIN_DAYS` çelişkisi → ÇÖZÜLDÜ

`src/config.py:TRAIN_DAYS = 1000` **canlı yolda kullanılmıyor**.

Kanıt: `scripts/morning_run.py:128` → `EN.TRAIN_DAYS`, `src/ensemble_morning.py:31` → `TRAIN_DAYS = 2000`.

**Canlıda etkin değer 2000 gün.** `config.TRAIN_DAYS=1000` ölü sabit (başka bir kod yoluna ait, güneş/rüzgarda da kullanılmıyor). Karışıklık kaynağı: iki ayrı sabit, aynı isim.

### P1.0b · `SHIFT_H = -1` — **GERÇEK BULGU**

`scripts/solar_mnwp.py:187`:
```python
for _nm in ("ghi","ghi2","cloud","cloud2","temp","temp2","wspd","wspd2","rh","rh2","pr","pr2"):
    globals()[_nm] = _s.shift(SHIFT_H)   # SHIFT_H = -1
```
`scripts/solar_live.py:66` ve `:91` de **tüm hava çerçevesini** aynı şekilde kaydırıyor.

**12 değişkenin hepsi kaydırılıyor.** Kullanıcının şüphesi doğru:

| Değişken tipi | Open-Meteo semantiği | −1 kaydırma doğru mu |
|---|---|---|
| `shortwave_radiation`, `direct/diffuse_radiation`, `precipitation` | **Aralık ortalaması** (saat sonunda etiketli) | ✅ **Doğru** |
| `temperature_2m`, `cloud_cover`, `wind_speed`, `relative_humidity` | **Anlık** (saatin başındaki değer) | ❌ **Yanlış** — 1 saat ileriye kayıyor |

**Etki:** Işınım için yapılan düzeltme fiziksel olarak doğru (EPİAŞ saati h = h:00–h+1:00 aralığı). Ama sıcaklık/bulut gibi anlık değişkenler gereksiz yere 1 saat ileri alınıyor. Güneş modelinde sıcaklık panel verimini etkiliyor → küçük ama sistematik hata. Yük modeli bu hatadan muaf (ayrı hava hattı kullanıyor).

**Bu bulgu bir tuning değil, hizalama hatası.** P1.3'te aynen korunuyor (baseline mevcut kodun dürüst ölçümü olmalı), düzeltme ayrı bir hipotez olarak listelenecek.

### P1.0c · Hedef saat hizalaması

| Taraf | Konvansiyon | Durum |
|---|---|---|
| EPİAŞ `rt_cons` / `wind` / `sun` | Saat `h` = **h:00–h+1:00 ortalaması** | Kaynak dokümanı doğrulanmalı → AÇIK SORULAR #1 |
| Open-Meteo ışınım | Saat `h` = **(h-1):00–h:00 ortalaması** (aralık sonunda etiketli) | −1 kaydırma bunu hizalar ✅ |
| Open-Meteo sıcaklık/bulut/rüzgar | Saat `h` = **anlık değer** | −1 kaydırma **bozuyor** ❌ |

### P1.0d · `SOLAR_CAPACITY_MW = 27445`

**ML yolunda kullanılmıyor.** Kullanıldığı yerler:
- `calibrate_solar_pr.py` — PR kalibrasyonu (fizik modeli)
- `morning_ptf.py` — fizik fallback (güneş ML kapsaması yetersizse)
- `morning_run.py` — fizik fallback

Yani ML modeli kapasite normalizasyonunu `rolling_30d_p99` zarfıyla yapıyor; 27.445 sabiti yalnız yedek fizik yolu için.

Kullanıcının gözlemi doğru: `sun` serisi tepesi ~4,2 GW → **27.445 MW lisanslı kurulu gücün çok üstünde**. Sabit, fizik fallback'ini 6-7 kat şişiriyor (kontratta 11.138 MW gibi değerler buradan geliyor). AÇIK SORULAR #2.

### P1.0e · Yük modelinde ışınım/bulut

`src/morning.py` yük feature'ları **`shortwave_radiation` ve `cloud_cover` içeriyor** (10 il nüfus-ağırlıklı, aynı hava hattı).

Katkı ağırlığı **ölçülmedi** (ablation yapılmadı — bu turda tuning yasak). Yük lag-baskın olduğu için ışınımın marjinal katkısı küçük olması beklenir, ama bu bir beklenti, ölçüm değil → AÇIK SORULAR #3.

---

## P1.1 — LEAKAGE TEYİDİ · **SONUÇ: ARŞİV SIZINTILI**

**Yöntem:** `multinwp/wind/Balikesir_2026Q4.parquet` içindeki `wind_speed_100m_ecmwf_ifs025` değerleri, Open-Meteo **Previous Runs API**'den aynı koordinat (39.65, 27.89) için çekilen `current` / `previous_day1` / `previous_day2` / `previous_day3` ile karşılaştırıldı. Ortak pencere: 2026-10-01 → 2026-11-15, **1.008 saat**.

| Karşılaştırma | Ortalama mutlak fark | Birebir eşit | Medyan |
|---|---|---|---|
| **Güncel run (en kısa lead)** | **3,30 km/h** | %2,8 | 2,35 |
| previous_day1 | 3,60 km/h | %3,0 | 2,50 |
| previous_day2 | 3,90 km/h | %0,9 | 2,90 |
| previous_day3 | 4,23 km/h | %1,7 | 3,00 |

**Yorum:** Farklar monoton artıyor — arşiv **en kısa lead'e en yakın**. Hiçbir vintage ile birebir eşleşmiyor (%3 eşitlik rastgele seviye), yani arşiv bu endpoint'lerin birebir kopyası değil; farklı bir kaynaktan/çalıştırmadan geliyor. Ama sıralama net: **arşiv, gün-öncesi tahminden daha iyi bir tahmin taşıyor.**

**Hüküm:** Rüzgar (ve aynı boru hattını kullanan güneş) backtest'i **iyimser**. "%15,53" ve "%10,76" gerçek gün-öncesi vintage ile **daha kötü** çıkacak. Ne kadar kötü — P1.3 ölçecek.

**Kontrol:** day1 vs day2 ortalama farkı 2,11 km/h, birebir eşitlik %3,0 → vintage'lar gerçekten ayrı, test ayırt edici güce sahip.

---

## P1.2 — HONEST ARŞİV

**Kaynak:** `https://previous-runs-api.open-meteo.com/v1/forecast`
**Form:** `hourly={var}_previous_day1,{var}_previous_day2` + `models=ecmwf_ifs025,gfs_seamless,icon_seamless`
**Dönen kolonlar:** `{var}_previous_day1_{model}`, `{var}_previous_day2_{model}`
**Dönem:** 2024-01-01 → bugün · **Path:** `data/weather/honest/{wind,solar}/<Il>_<YYYY>Q<n>.parquet`

| Set | İl | Değişken | Kolon/istek |
|---|---|---|---|
| Rüzgar | 10 (Balıkesir, İzmir, Çanakkale, Manisa, Hatay, Aydın, İstanbul, Kırklareli, Afyon, Kayseri) | wind_speed_100m, wind_direction_100m, wind_speed_10m, temperature_2m, boundary_layer_height | 15 → 30 (3 model × 2 vintage) |
| Güneş | 8 (Konya, Şanlıurfa, Ankara, Kayseri, İzmir, Antalya, Mersin, Kahramanmaraş) | shortwave_radiation, direct_radiation, diffuse_radiation, cloud_cover_low/mid/high, temperature_2m | 14 → 42 |

Durum: **çekim sürüyor** (rüzgar 10 il × 23 ay × 5 değişken ayrı istek; güneş 8 il).

### P1.2 bulguları — API kısıtları (ölçüldü)

| Bulgu | Kanıt |
|---|---|
| **`cloud_cover_low/mid/high` Previous Runs API'de YOK** | 3 modelde de **%0 doluluk**. Toplam `cloud_cover` kullanılmalı (%91-100 dolu) |
| **Rüzgar değişkenleri tek istekte çakışıyor** | 2+ değişken birlikte → HTTP 400 `SurfacePressureAndHeightVariable`. Tek tek → 200. Çözüm: değişken başına istek |
| **`{var}_previous_dayN_{model}` kolon adı** | Model adı **değişkene yazılmaz**; `models=` parametresine verilir. Yanlış yazım 400 verir |
| **2024 kapsamı eksik** | 2024: %82-91 · **2025: %100** · 2026: %100 |

**Karar:** dürüst baseline'ın değerlendirme penceresi **2025-01-01 → bugün** (kapsam tam). 2024 yalnız ısınma/eğitim için, eksik.

### Çekilen arşiv (durum)

| Set | İl | Dönem | Kolon |
|---|---|---|---|
| Güneş | 8 | 2024-01-01 → 2026-11-15 | 42 → **30** (low/mid/high çıkarıldı, `cloud_cover` eklendi) |
| Rüzgar | 10 | 2024-01-01 → 2026-11-15 | 30 (5 değişken × 3 model × 2 vintage) |

Her il 25.200 saat (2024-01-01 → 2026-11-15).

---

## P1.3 — HONEST BASELINE

Bekliyor (P1.2 çekimi bitince).

---

## P1.4 — ÖLÇÜM

Bekliyor.

---

## AÇIK SORULAR

1. **EPİAŞ saat konvansiyonu** — `rt_cons`/`wind`/`sun` saat `h` gerçekten h:00–h+1:00 ortalaması mı? Kaynak dokümanı bulunamadı; varsayım olarak kullanılıyor.
2. **`SOLAR_CAPACITY_MW = 27445`** — hangi kaynaktan, hangi tarihte? `sun` serisi tepesi ~4,2 GW iken bu sabit 27,4 GW; fizik fallback'i 6-7 kat şişiriyor. Doğrusu ne olmalı?
3. **Yük modelinde ışınım/bulut katkısı** — ablation yapılmadı (bu turda tuning yasak).
4. **`SHIFT_H=-1` düzeltmesinin büyüklüğü** — anlık değişkenlere uygulanması ne kadar hata ekliyor? Ölçülmedi.
5. **Arşiv neden hiçbir vintage ile birebir eşleşmiyor** — hangi endpoint/kaynaktan geliyor? `pull_multinwp.py` geçmiş tarih için `historical-forecast-api` kullanıyordu; o endpoint'in default lead'i en kısa olan.
6. **`boundary_layer_height`** — 3 modelde de mevcut mu, yoksa bazı modellerde NaN mı dönüyor? (test 15 kolon döndü, doluluk kontrol edilmedi)
