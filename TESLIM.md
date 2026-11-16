# TESLİM — Türkiye T+1/T+2 Talep, Güneş, Rüzgar Tahmin Sistemi

**Devralan:** Ali Bey
**Teslim tarihi:** 2026-11-15
**Teslim eden:** Çağatay Bayrak

---

## 1. BU NE

Her sabah **09:00'da** otomatik çalışan bir tahmin sistemi. Üç seri üretir:

| Seri | Ne | Ufuk |
|---|---|---|
| **Talep** | Türkiye ulusal elektrik tüketimi | T+1 (yarın) + T+2 (öbür gün) |
| **Güneş** | Ulusal güneş üretimi | T+1 + T+2 |
| **Rüzgar** | Ulusal rüzgar üretimi | T+1 + T+2 |

Çıktı PTF fiyat modeline **feature** olarak girer. PTF modeli bu projenin kapsamında değildir.

**Ürün dosyası:** `data/forecast/ptf_features/latest.parquet` — 48 satır (24 saat × 2 ufuk) × 13 kolon.

---

## 2. ŞU AN NE ÇALIŞIYOR

| Bileşen | Durum |
|---|---|
| Zamanlanmış görev | `TR_Morning_Yarim_Tahmin`, her gün 09:00, SYSTEM |
| Zincir | `run_morning.bat` |
| Çalıştığı makine | `C:\projeler\turkiye-t2-daysahead` (Tailscale: 100.124.58.73) |
| Son kontrat | 2026-11-15 09:05, 48×13, denge 0.003906 MW |
| Hata kaydı | `data/logs/LAST_FAILURE.txt` — **boşsa iyi** |

### Zincir sırası (`run_morning.bat`)

```
1. scripts/pull_daily.py          hava çekimi (rüzgar 10 il + yük 10 il)
2. scripts/morning_run.py         talep tahmini → yarin_YYYY-MM-DD.parquet
3. scripts/morning_scoreboard.py  skor defteri (gerçekleşen + TEİAŞ)
4. scripts/wind_live_v5.py        rüzgar → wind_live.parquet
5. scripts/solar_live.py          güneş → solar_live.parquet
6. scripts/morning_ptf.py         48×13 kontrat → latest.parquet
7. scripts/archive_contract.py    arşiv + sidecar
8. push_morning.bat               GitHub'a push
```

> **KRİTİK:** `morning_ptf.py` 4 ve 5'in çıktısını okur. **Tek başına çalıştırılırsa bayat veriyle tutarsız kontrat üretir.** Her zaman tam zinciri çalıştır.

---

## 3. SKORLAR — DÜRÜST TABLO

Ölçüm: 365 gün walk-forward, canlı-eşdeğer (karar günü verisi 07:00'de kesik).

| Seri | Ufuk | WAPE | nMAE | Not |
|---|---|---|---|---|
| **Talep** | T+1 | **%1,66** | 684 MW | son 90 gün %1,51 |
| **Talep** | T+2 | **%1,82** | 875 MW | per-horizon eğitim |
| **Güneş** | T+1 | **%10,76** | 109 MW | gündüz saatleri |
| **Güneş** | T+2 | **%10,86** | 110 MW | gündüz saatleri |
| **Rüzgar** | T+1 | **%15,53** | 711 MW | ⚠️ aşağıya bak |
| **Rüzgar** | T+2 | **%15,63** | 716 MW | ⚠️ aşağıya bak |

**TEİAŞ referansı:** %1,62 (8 gün, gün-öncesi plan). Talep T+1 bu seviyede.

### ⚠️ Bilmen gereken üç uyarı

**1. Rüzgar/güneş backtest'i iyimser.** Eğitim havası `historical-forecast-api`'den geliyor ve bu endpoint **o saat için en kısa lead'li** tahmini döndürüyor — gerçek gün-öncesi vintage değil. Üstelik aynı sorgu bugün farklı değer veriyor (API geçmişi revize ediyor) → **arşiv tekrar üretilemez.**

**2. Talep backtest'i adil.** `lead_weather` arşivi `_previous_day1/_previous_day2` kullanıyor = "1/2 gün önce yayınlanan tahmin". Reanalysis (ERA5) bilinçli olarak dışlandı.

**3. Canlı skor henüz birikmedi.** Sistem 15 Kasım'dan itibaren yeni kodla çalışıyor; önceki günler eski kodla geçti. Anlamlı canlı ölçüm için **~30 gün** gerekiyor.

---

## 4. KURULUM

### 4.1 Gerekenler
- Windows + Python 3.11 (`C:\Users\<kullanıcı>\AppData\Local\Programs\Python\Python311\python.exe`)
- Tailscale (uzak makineye erişim için)
- EPİAŞ şeffaflık hesabı

> **Python 3.14 kullanma** — LGBM/torch `access violation` ile çöküyor. 3.11 zorunlu.

### 4.2 Kod
```bash
git clone https://github.com/dattik12/turkiye-t2-daysahead.git
```

> Repo **public**. `.env` repoda **yok**, ayrıca verilecek.

### 4.3 Kimlik bilgileri
`.env` dosyasını proje köküne koy:
```
EPTR_USERNAME=<EPİAŞ kullanıcı adı>
EPTR_PASSWORD=<EPİAŞ şifre>
```
`.gitignore` `.env` içeriyor — commit edilmez.

### 4.4 Veri dizinleri

| Yol | İçerik | Boyut |
|---|---|---|
| `C:\projeler\multinwp\wind` | NWP rüzgar arşivi (10 il × 3 model) | ~160 dosya |
| `C:\projeler\multinwp\solar` | NWP güneş arşivi | ~128 dosya |
| `C:\projeler\lead_weather` | Yük modeli hava arşivi (10 il × 7 değişken) | 10 dosya |
| `C:\projeler\prov_lead100` | 39 il 100m rüzgar (eski) | 624 dosya |
| `data\raw\epias_national_rt_gen_2021_2026.parquet` | EPİAŞ ulusal üretim gerçekleşen | — |
| `data\forecast\weather_consumed\` | Her koşuda tüketilen hava (denetim izi) | günlük |
| `data\weather\honest\` | Previous Runs API vintage arşivi (Phase 1) | — |

### 4.5 Zamanlanmış görev
```cmd
schtasks /create /tn TR_Morning_Yarim_Tahmin /tr "C:\projeler\turkiye-t2-daysahead\run_morning.bat" /sc daily /st 09:00 /ru SYSTEM
```

---

## 5. KOD DÜZENİ

`scripts/README.md` tam envanter veriyor. Özet:

| Grup | Sayı | Not |
|---|---|---|
| **Üretim** | 7 | `run_morning.bat` çağırır |
| Doğrulama | 3 | `verify_contract.py` 17 kapı |
| Ölçüm/tanı | 7 | `load_audit2.py`, `wind_errstruct.py` vb. |
| Veri çekimi | 7+ | tek seferlik arşiv kurma |
| Araştırma | ~55 | kapalı konular işaretli |

### Model mimarileri

| Model | Mimari | Eğitim penceresi |
|---|---|---|
| Talep | LGBM (0.10) + CatBoost (0.20) + MLP (0.70), 3 tohum `[7,17,27]` | **2000 gün** |
| Güneş | LGBM, CF-normalize | 540 gün |
| Rüzgar | LGBM, IEC power curve, CF-normalize | 365 gün |

---

## 6. AÇIK İŞLER

### Yüksek öncelik
1. **Rüzgar/güneş dürüst backtest** — Phase 1 yarım. Arşiv %75-77 dolu, kota yarın açılıyor. `PHASE1.md` oku.
2. **Canlı skor birikimi** — 30 gün bekle, sonra gerçek skoru ölç.
3. **`SHIFT_H = -1` hatası** — `solar_mnwp.py:187` 12 değişkenin hepsini kaydırıyor. Işınım için doğru, sıcaklık/bulut için yanlış. **Phase 2 deney #1.**

### Orta öncelik
4. `SOLAR_CAPACITY_MW = 27445` yanlış (gerçek ~4.200 MW) — fizik fallback'i 6-7 kat şişiriyor.
5. Rüzgar il ağırlıkları düz ortalama — kurulu güce göre olmalı.
6. Kurulu kapasite serisi yok — `rolling_30d_p99` vekil kullanılıyor.

### Bilinen tuzaklar
- **Open-Meteo kotası** — istemci tarafı sayaç şart. `scripts/openmeteo_client.py` (MRC'den alındı) 429 sarmalını çözüyor. **İki çekimi paralel koşturma** — kotalar çakışır.
- **Windows'ta `tail` yok** — `powershell Get-Content -Tail` kullan.
- **SSH ile çok satırlı `python -c`** — `\n` kaçışı çalışmıyor. Script dosyası yaz, `scp` ile at, uzakta koş.
- **`archive/ptf_*.parquet` boş** — arşiv yolu farklı olabilir, doğrulanmadı.

---

## 7. BELGELER

| Dosya | İçerik |
|---|---|
| `AUDIT.md` | Phase 0 — üç modülün tam denetimi, veri soy ağacı, hedef kalitesi, hipotez listesi, 12 açık soru |
| `PHASE1.md` | Phase 1 — dürüst baseline çalışması, leakage kanıtı, API kısıtları |
| `docs/PHASE_LOG.md` | Kronolojik deney kaydı — ne denendi, ne kazandı, ne kaybetti |
| `scripts/README.md` | Script envanteri, hangisi üretim hangisi araştırma |

---

## 8. İLK GÜN YAPILACAKLAR

1. Repoyu klonla, `.env`'i yerleştir
2. `python -c "import lightgbm, catboost, torch; print('ok')"` — Python 3.11 ile
3. Veri dizinlerini kontrol et (`C:\projeler\multinwp` vb.)
4. `run_morning.bat`'i elle bir kez çalıştır, `data/logs/` çıktısını oku
5. `scripts/verify_contract.py` koş — 17 kapıdan geçmeli
6. `AUDIT.md` ve `PHASE1.md`'i oku

---

## İLETİŞİM

Çağatay Bayrak · hcagataybayrak@gmail.com
