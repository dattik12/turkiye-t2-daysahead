import sys
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
import pandas as pd
from src import data as D

c = D.load_or_create_consumption()
print("kolonlar:", list(c.columns))
for col in c.columns:
    s = c[col]
    print(f"  {col}: dolu={int(s.notna().sum())} | {s.first_valid_index()} -> {s.last_valid_index()}")
if "load_plan" in c.columns:
    lp = c["load_plan"].dropna()
    print("\nload_plan ornek:")
    print(lp.head(3))
    print("gun sayisi:", lp.groupby(lp.index.normalize()).size().shape[0])
