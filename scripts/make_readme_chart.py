# -*- coding: utf-8 -*-
"""README için performans grafiği üretir -> docs/performance.png

Kaynak: data/exports/backtest_t1_t2_2025_2026.csv (saatlik backtest çıktısı).
Yeniden üretim: python -m scripts.make_readme_chart
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, 'data', 'exports', 'backtest_t1_t2_2025_2026.csv')
OUT = os.path.join(ROOT, 'docs', 'performance.png')

INK = '#9AA4B2'      # hem açık hem koyu temada okunur
BLUE = '#4C9BE8'     # T+1
ORANGE = '#F5A742'   # T+2
GRAY = '#7A8699'     # TEİAŞ


def main():
    df = pd.read_csv(CSV, parse_dates=['zaman_utc3'])
    df['ay'] = df['zaman_utc3'].dt.to_period('M')

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    fig.patch.set_alpha(0)

    # --- Panel 1: aylık MAPE barları ---
    ax = axes[0]
    m = df.groupby('ay')[['t1_mape_pct', 't2_mape_pct', 'teias_plan_mw']].agg(
        t1=('t1_mape_pct', 'mean'))
    g = df.groupby('ay').agg(t1=('t1_mape_pct', 'mean'),
                             t2=('t2_mape_pct', 'mean'))
    tp = df.dropna(subset=['teias_plan_mw']).copy()
    tp['teias_mape'] = ((tp['teias_plan_mw'] - tp['gerceklesen_mw']).abs()
                        / tp['gerceklesen_mw'] * 100)
    teias = tp.groupby('ay')['teias_mape'].mean()

    x = range(len(g))
    w = 0.27
    ax.bar([i - w for i in x], g['t1'], width=w, color=BLUE, label='T+1 (bizim)')
    ax.bar(list(x), g['t2'], width=w, color=ORANGE, label='T+2 (bizim)')
    teias_aligned = teias.reindex(g.index).fillna(0.0)
    ax.bar([i + w for i in x], teias_aligned.values, width=w,
           color=GRAY, label='TEİAŞ plan (T+1)')
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(i)[2:] for i in g.index], rotation=45, ha='right',
                       fontsize=8, color=INK)
    ax.set_ylabel('MAPE (%)', color=INK)
    ax.tick_params(colors=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, labelcolor=INK, fontsize=8, loc='upper left')
    ax.set_title('Aylık ortalama MAPE — 2025 Ocak → 2026 Ağustos',
                 color=INK, fontsize=10)

    # --- Panel 2: örnek hafta (son tam hafta) ---
    ax = axes[1]
    end = df['zaman_utc3'].max().normalize() - pd.Timedelta(days=7)
    start = end - pd.Timedelta(days=6)
    wk = df[(df['zaman_utc3'] >= start) & (df['zaman_utc3'] < end + pd.Timedelta(days=1))]
    ax.plot(wk['zaman_utc3'], wk['gerceklesen_mw'] / 1000, color='#D9DEE7',
            lw=1.8, label='Gerçekleşen')
    ax.plot(wk['zaman_utc3'], wk['t1_forecast_mw'] / 1000, color=BLUE,
            lw=1.4, ls='--', label='T+1 tahmin')
    ax.plot(wk['zaman_utc3'], wk['teias_plan_mw'] / 1000, color=GRAY,
            lw=1.2, ls=':', label='TEİAŞ plan')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
    ax.tick_params(colors=INK, labelsize=8)
    ax.set_ylabel('GW', color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, labelcolor=INK, fontsize=8, loc='upper left')
    wk_mape = wk['t1_mape_pct'].mean()
    ax.set_title('Örnek hafta: %s – %s (T+1 MAPE %%%.2f)'
                 % (start.strftime('%d.%m'), end.strftime('%d.%m'), wk_mape),
                 color=INK, fontsize=10)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, transparent=True)
    print('yazıldı:', OUT)

    # Konsola özet istatistikler (README sayıları için kanıt)
    n = len(df)
    print('satır:', n)
    print('T+1 saatlik MAPE ort: %.2f' % df['t1_mape_pct'].mean())
    print('T+2 saatlik MAPE ort: %.2f' % df['t2_mape_pct'].mean())
    print('TEİAŞ plan MAPE: %.2f' % tp['teias_mape'].mean())
    print('|hata|<%%2 payı T+1: %.1f' % (100 * (df['t1_mape_pct'] < 2).mean()))
    print('|hata|<%%3 payı T+2: %.1f' % (100 * (df['t2_mape_pct'] < 3).mean()))


if __name__ == '__main__':
    sys.exit(main())
