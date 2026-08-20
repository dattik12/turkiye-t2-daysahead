# EDA: Bayram / Ramazan / Bridge-Day Segment Bazlı Hata Analizi

**Kaynak:** `data/results/backtest_sim_2025_2026.csv` — 593 karar günü, 594 hedef gün, **28 464** tahmin satırı (H1+H2), dt aralığı `2025-01-02 .. 2026-08-18`. Tüm sayılar backtest’ten hesaplanmıştır (uydurma yok). Takvim `src/features.py` → `HOLIDAY_DATES` (dini), `OFFICIAL` (resmi), `RAMADAN` ile birebir.

**Metrik:** `AE = |pred - actual|`, `APE = AE/actual*100`, `MAPE = mean(APE)`, `MAE = mean(AE)`. Rapor **tahmin-satırı** (her `decision_date × dt × horizon`) seviyesindedir; her `dt` günü 48 satır içerir (H1+H2 örtüşmesi), 2025-01-02 ve 2026-08-18 kenar günler 24 satır.

---

## 1) Özet — Kritik Bulgular

| Soru | Cevap (backtest) |
|---|---|
| **Bayram günü MAPE** | **%6.49** (672 satır, n=14 günün 48×2 horizon'ı; H1 %5.50 / H2 %7.48) — ortalamanın **3.3×** üstü |
| **Arife (bayram-1) MAPE** | **%7.00** — **en kötü segment**; ort +5.05 pp, tüm hatanın fazlasının %~35’i buradan |
| **Arefe-1 (bayram-2) MAPE** | **%6.21** |
| **Bayram+1 (after) MAPE** | **%5.86** |
| **Resmi tatil (1 Oca, 23 Nis vb) MAPE** | **%2.75** (+0.80 pp) — ılımlı |
| **Bridge-day MAPE** | **%1.88** (240 satır, 5 gün) — **ortalamadan düşük** (−0.07 pp); model köprü günleri iyi genelleştiriyor |
| **Ramazan ayı MAPE** | **%2.43** (+0.48 pp) ; bayram hariç **%2.23** (+0.28 pp) → Ramazan’ın kendisi ılımlı, asıl darbe bayram/arife |
| **Normal gün (strict) MAPE** | **%1.72** (23 952 satır, %84 pay) — fırsat: model normal günlerde zaten iyi |
| **Horizon etkisi** | H1 %1.63 / H2 %2.27 (+0.64 pp); bayramda makas **+1.98 pp** (H2 çok daha kötü) |
| **Saat dilimi** | Gündüz 10-16 %2.31 (en kötü) > sabah 06-09 %2.06 > akşam pik %1.89 > gece 00-05 %1.55 (en iyi) |
| **Hafta içi vs hafta sonu** | Fark yok: içi %1.95 / sonu %1.96 |

**Hangi segmentler ortalamayı yukarı çekiyor?** Sırayla: `arife > bayram+arife > dini bayram > arefe-1 > bayram+1 > tatil-geniş`. Bu 6 segment toplam satırın %~6.4’ü ama toplam APE fazlasının (excess) %~80’ini üretiyor. Tersine `bridge-day`, `hafta sonu/içi`, `gece` ortalamayı **aşağı** çekiyor.

> **Leak notu:** `src/features.py:make_row` training ve inference için aynı fonksiyonu kullanır; tüm lag/rolling feature’lar `shift` ile hedef saatten önce kalır. Bu EDA sadece gerçekleşen `actual_mw` ile `pred_mw` hatasını ölçer, feature leak’i ölçmez.

---

## 2) Segment Tablosu (MAPE’ye göre azalan)

Kaynak CSV: `data/results/eda_segments.csv` — aynı tabloyu makine-okunur verir. `share_%` = satır payı, `delta_vs_all_pp` = ALL’e göre fark (pp).

| segment | n | MAPE % | MAE MW | RMSE MW | p50 | p90 | p95 | share_% | Δ vs ALL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arife | 672 | 7.00 | 2080 | 2576 | 6.11 | 13.63 | 16.27 | 2.36 | +5.05 |
| bayram+arife | 864 | 6.83 | 2006 | 2536 | 5.66 | 13.99 | 16.46 | 3.04 | +4.88 |
| dini bayram | 672 | 6.49 | 1819 | 2343 | 5.13 | 13.97 | 16.89 | 2.36 | +4.54 |
| arefe-1 (bayram-2) | 672 | 6.21 | 1945 | 2515 | 5.09 | 13.49 | 15.80 | 2.36 | +4.26 |
| bayram+1 (after) | 672 | 5.86 | 1755 | 2379 | 4.60 | 13.23 | 15.85 | 2.36 | +3.91 |
| tatil geniş (bayram+resmi+arife+after+bridge) | 1824 | 4.83 | 1541 | 2124 | 3.30 | 11.64 | 14.49 | 6.41 | +2.88 |
| resmi tatil | 528 | 2.75 | 1012 | 1416 | 2.13 | 5.70 | 7.39 | 1.86 | +0.80 |
| Ramazan ayı | 2880 | 2.43 | 929 | 1297 | 1.76 | 5.16 | 7.26 | 10.12 | +0.48 |
| gündüz 10-16 | 8302 | 2.31 | 987 | 1349 | 1.69 | 4.93 | 6.40 | 29.17 | +0.36 |
| H2 | 14232 | 2.27 | 910 | 1257 | 1.69 | 4.72 | 6.24 | 50.00 | +0.32 |
| Ramazan (bayram hariç) | 2784 | 2.23 | 878 | 1207 | 1.71 | 4.67 | 6.28 | 9.78 | +0.28 |
| sabah ramp 06-09 | 4744 | 2.06 | 754 | 1100 | 1.42 | 4.34 | 5.62 | 16.67 | +0.11 |
| hafta sonu | 8160 | 1.96 | 728 | 996 | 1.42 | 4.25 | 5.57 | 28.67 | +0.01 |
| **ALL** | **28464** | **1.95** | **780** | **1104** | **1.39** | **4.14** | **5.54** | 100.00 | 0.00 |
| hafta içi | 20304 | 1.95 | 800 | 1145 | 1.37 | 4.09 | 5.52 | 71.33 | −0.00 |
| akşam pik 17-22 | 7116 | 1.89 | 818 | 1111 | 1.41 | 3.97 | 5.13 | 25.00 | −0.06 |
| bridge-day | 240 | 1.88 | 769 | 1005 | 1.45 | 4.19 | 5.34 | 0.84 | −0.07 |
| normal (loose) | 27264 | 1.82 | 749 | 1048 | 1.35 | 3.93 | 5.05 | 95.78 | −0.13 |
| gece 23 | 1186 | 1.80 | 715 | 964 | 1.38 | 3.77 | 4.80 | 4.17 | −0.15 |
| tatil dışı hafta sonu | 6960 | 1.75 | 676 | 908 | 1.34 | 3.87 | 4.82 | 24.45 | −0.20 |
| normal (strict) | 23952 | 1.72 | 718 | 989 | 1.30 | 3.74 | 4.73 | 84.15 | −0.23 |
| tatil dışı hafta içi | 16992 | 1.71 | 735 | 1020 | 1.29 | 3.68 | 4.69 | 59.70 | −0.24 |
| H1 | 14232 | 1.63 | 649 | 927 | 1.16 | 3.48 | 4.58 | 50.00 | −0.32 |
| gece 00-05 | 7116 | 1.55 | 528 | 749 | 1.09 | 3.37 | 4.43 | 25.00 | −0.41 |

**Okuma:** `normal (strict)` = dini+resmi+arife+after+bridge+ramazan dışı; `normal (loose)` = sadece dini+resmi hariç (arife/bridge dahil).

---

## 3) Horizon Kırılımı — Kritik Tatil Segmentlerinde H2 Bozulması

| segment | H1 MAPE | H2 MAPE | H2−H1 |
|---|---:|---:|---:|
| dini bayram | 5.50 | 7.48 | **+1.98** |
| arife | 6.28 | 7.72 | +1.44 |
| bayram+1 | 5.21 | 6.52 | +1.31 |
| Ramazan ayı | 2.03 | 2.83 | +0.80 |
| bridge-day | 1.57 | 2.19 | +0.62 |
| resmi tatil | 2.95 | 2.55 | −0.39 (ters: H2 daha iyi) |

Yorum: dini tatillerde H2 hatası belirgin sıçrıyor — model 2 gün önden bayram şeklini yakalayamıyor. Resmi tatillerde ise H2 daha iyi (tesadüf değil: resmi tatiller tek gün, H2’nin gördüğü lag’ler daha stabil).

---

## 4) En Kötü 20 Gün (günlük MAPE, H1+H2 ortalaması; 48 tahmin/gün)

`data/results/eda_daily.csv` tam liste (594 gün). İlk 20:

| # | tarih | gün | MAPE | MAE | flags |
|---|---|---|---|---|---|
| 1 | 2025-04-01 | Salı | 9.62 | 2580 | bayram,after |
| 2 | 2026-05-25 | Pzt | 9.52 | 3250 | arefe-1 |
| 3 | 2025-06-05 | Per | 9.21 | 2903 | arife,arefe-1 |
| 4 | 2026-03-20 | Cum | 9.01 | 2624 | bayram,arife,after |
| 5 | 2026-03-18 | Çar | 9.01 | 3607 | arife,arefe-1,ramazan |
| 6 | 2025-06-06 | Cum | 8.80 | 2249 | bayram,arife,arefe-1 |
| 7 | 2025-06-10 | Sal | 8.68 | 3545 | after |
| 8 | 2025-03-30 | Paz | 8.12 | 2086 | bayram,arife,arefe-1,ramazan |
| 9 | 2026-03-19 | Per | 7.96 | 2736 | bayram,arife,arefe-1,ramazan |
| 10 | 2025-03-31 | Pzt | 7.41 | 1917 | bayram,arife,after |
| 11 | 2026-05-26 | Sal | 7.41 | 2203 | arife,arefe-1 |
| 12 | 2025-06-08 | Paz | 6.71 | 1941 | bayram,arife,after |
| 13 | 2025-06-09 | Pzt | 6.65 | 1923 | bayram,after |
| 14 | 2026-03-23 | Pzt | 6.62 | 2844 | normal* |
| 15 | 2025-03-29 | Cmt | 6.41 | 1943 | arife,arefe-1,ramazan |
| 16 | 2026-03-21 | Cmt | 6.23 | 1816 | bayram,after |
| 17 | 2026-01-01 | Per | 6.21 | 2205 | resmi |
| 18 | 2025-06-07 | Cmt | 6.17 | 1743 | bayram,arife,after,arefe-1 |
| 19 | 2025-04-02 | Çar | 5.32 | 1866 | after |
| 20 | 2025-04-03 | Per | 5.25 | 1836 | normal* |

`*` 2026-03-23 ve 2025-04-03 bayrama yakın günler (Ramazan bayramı sonrası +2/+3) ama mevcut `arife/after/arefe-1` tanımında “normal” görünüyor — **bayram kuyruğu** etkisi; bölüm 6’da önerilen genişletilmiş pencere ile yakalanır.

---

## 5) En Kötü 20 Saat (tekil `dt` tahmini; `data/results/eda_worst_hours.csv`)

| dt | APE | AE | pred | actual | bağlam |
|---|---:|---:|---:|---:|---|
| 2025-06-09 08:00 | 37.5 | 9769 | 35846 | 26077 | bayram+after (Kurban) |
| 2025-04-01 08:00 | 31.7 | 7484 | 31057 | 23574 | bayram+after (Ramazan) |
| 2025-04-01 09:00 | 28.3 | 7312 | 33145 | 25832 | bayram+after |
| 2026-01-01 09:00 | 26.0 | 8404 | 40722 | 32317 | resmi (yılbaşı) |
| 2026-05-26 08:00 | 24.8 | 6690 | 33681 | 26991 | arife |
| 2026-03-20 15:00 | 23.7 | 6706 | 35011 | 28304 | bayram+arife+after çakışma |
| 2026-03-20 14:00 | 23.2 | 6616 | 35135 | 28520 | bayram+arife+after |
| 2025-06-06 09:00 | 22.1 | 5467 | 30167 | 24699 | bayram+arife |
| 2026-05-26 09:00 | 21.8 | 6366 | 35504 | 29137 | arife |
| 2025-06-09 08:00* | 21.8 | 5674 | 31750 | 26077 | bayram+after (H1/H2 iki tahmin; aynı saat iki kez listede) |

Desen: **sabah 07-11** saatleri en büyük APE’yi veriyor — arife/bayram sabah ramp’ı modelin en zayıf yeri.

---

## 6) Bridge-Day Analizi

### 6.1 Tespit edilen köprü günler (backtest aralığında)

| tarih | gün | komşular | günlük MAPE | MAE |
|---|---|---|---|---|
| 2025-05-02 | Cum | Per 01 May (resmi) – Cmt hafta sonu | 2.69 | 964 |
| 2025-07-14 | Pzt | Cmt-Paz hafta sonu – Sal 15 Tem (resmi) | 1.33 | 645 |
| 2026-01-02 | Cum | Per 01 Oca (resmi) – Cmt hafta sonu | 2.73 | 1218 |
| 2026-04-24 | Cum | Per 23 Nis (resmi) – Cmt hafta sonu | 1.50 | 572 |
| 2026-05-18 | Pzt | Cmt-Paz hafta sonu – Sal 19 May (resmi) | 1.14 | 446 |

Ortalama **%1.88** — normal günle aynı, resmi tatilden belirgin iyi. Köprü günler **sorun değil**.

### 6.2 Otomatik tespit mantığı (öneri)

> **Kural:** `D` günü **bridge-day** iff:
> 1. `D` hafta içi (Pzt–Cum) ve `D ∉ HOLIDAY_DATES ∪ OFFICIAL` ve `D` hafta sonu değil,
> 2. `(D−1 ∈ HOLIDAY_DATES ∪ OFFICIAL  ve  D+1 hafta sonu)` **veya** `(D−1 hafta sonu  ve  D+1 ∈ HOLIDAY_DATES ∪ OFFICIAL)`.
>
> Yani *resmi/dini tatil ile hafta sonu arasında kalan tek iş günü*. Tek gün boşluk şartı kritik — iki iş günü boşluk varsa köprü sayılmaz.

**Python (features.py’ye eklenebilir):**

```python
def bridge_days(holiday_dates: set, official_dates: set,
                start: str, end: str) -> set:
    hol = set(pd.to_datetime(list(holiday_dates|official_dates)).date)
    out = set()
    for d in pd.date_range(start, end, freq="D"):
        dd = d.date()
        if dd in hol or dd.weekday() >= 5:
            continue
        prev = (d - pd.Timedelta(days=1)).date()
        nxt  = (d + pd.Timedelta(days=1)).date()
        prev_is_hol = prev in hol
        nxt_is_hol  = nxt in hol
        prev_is_we  = prev.weekday() >= 5
        nxt_is_we   = nxt.weekday() >= 5
        if (prev_is_hol and nxt_is_we) or (prev_is_we and nxt_is_hol):
            out.add(dd)
    return out
```

Notlar:
- Çok günlük bayram bloklarında kural blok **kenarlarında** çalışır (örn. 2025-03-30..04-01 blok → öncesi/sonrası köprü aranır).
- Tatil hafta sonuna değiyorsa boşluk 0 gün → köprü yok (doğru).
- İsteğe bağlı genişletme: `bridge-2` (tatil–hafta sonu arası 2 iş günü) ayrı feature olarak eklenebilir ama backtest’te 0 örnek var (2025-2026 takviminde yok).

**Üretim önerisi:** `is_bridge` binary feature’ı + `days_to_holiday/next_holiday` ile birlikte kullan; ayrı `bridge × hour` interaksiyonu eklemeye gerek yok (hata zaten düşük).

---

## 7) Ramazan Etkisi

- Ramazan ayı (30+30=60 gün, 2 880 satır) MAPE %2.43 — arife/bayram hariç %2.23. Yani Ramazan’ın kendisi **+0.28 pp** ek hata getiriyor, büyük değil.
- Ramazan içindeki en kötü gün 2026-03-18 %9.01 ve 2025-03-30 %8.12 — ikisi de arife/bayram ile çakışıyor; saf Ramazan günleri (örn. 2025-03-20 %4.42) daha ılımlı.
- İftar/sahur saatleri hipotezi için saat kırılımı: Ramazan günlerinde akşam pik (17-22) MAPE %2.1, gece 00-05 %1.6 — fark var ama küçük; ayrı saat×Ramazan feature’ı denenebilir ama öncelik değil.

---

## 8) Ne Yapılmalı (öncelik sırası)

1. **Arife / bayram-2 / bayram+1 penceresi** — en yüksek kaldıraç. `is_arife`, `is_after`, `arefe-1` zaten var ama yetmiyor. Öneri: `bayram_kuyruğu` feature’ı: `d ∈ {arife, bayram, bayram+1, bayram+2}` için ayrı seviye; özellikle `bayram+2` (2026-03-23, 2025-04-03 gibi) şu an “normal” görünüyor ve %5-6 hata veriyor → pencereyi +2 güne genişlet.
2. **H2 için ayrı model / düzeltme** — bayramda H2 hatası H1’den +2 pp fazla. H2’ye `pred_d1` zaten var ama yetersiz. Bayramda H2’yi H1’e daha agresif çek veya H2 için ayrı LightGBM (bayram ağırlıklı loss).
3. **Saat 08-11 düzeltmesi** — en kötü 20 saatin 14’ü 08-11 arası. `hour × is_holiday/arife` interaksiyonunu güçlendir (mevcut `commercial_CDD_x_workhour` yetersiz).
4. **Köprü gün** — dokunma; feature olarak `is_bridge` ekle yeter, model zaten iyi.
5. **Ramazan** — düşük öncelik; `is_ramadan × hour` interaksiyonu denenebilir ama kazanç sınırlı.

---

## 9) Dosyalar

- `data/results/eda_segments.csv` — segment tablosu (makine-okunur)
- `data/results/eda_daily.csv` — 594 günün günlük MAPE/MAE listesi
- `data/results/eda_worst_hours.csv` — en kötü 20 saat (APE’ye göre)
- Bu rapor: `data/results/eda_segments.md`

*Üretim:* `scripts/eda_segments.py` (`.venv` Python 3.11, `pandas`, `lightgbm` gerekmez) ile yeniden üretilebilir: `.\.venv\Scripts\python.exe scripts/eda_segments.py`

---

*Son güncelleme: backtest `2025-01-02 .. 2026-08-18` (28 464 satır). Tüm MAPE/MAE değerleri `pred_mw` vs `actual_mw`’den hesaplanmıştır.*
