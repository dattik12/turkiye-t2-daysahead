import os, sys, hashlib, importlib.util
import pandas as pd, numpy as np
os.chdir(r"C:\projeler\turkiye-t2-daysahead")
sys.path.insert(0, r"C:\projeler\turkiye-t2-daysahead")
from src import data as D, morning as MO

spec = importlib.util.spec_from_file_location("la2", r"scripts\load_audit2.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

cons = D.load_or_create_consumption()
y = cons["rt_cons"].dropna(); y.index = pd.to_datetime(y.index)
if getattr(y.index, "tz", None) is not None:
    y.index = y.index.tz_localize(None)
N1, N2 = m.build_nat(1), m.build_nat(2)

def build(NAT, off, dec):
    tgt = pd.date_range(dec + pd.Timedelta(days=off), periods=24, freq="h")
    try:
        return MO.build_features(y, NAT, tgt, decision_date=dec)
    except TypeError:
        return MO.build_features(y, NAT, tgt)

def hsh(X):
    return hashlib.md5(pd.util.hash_pandas_object(X.fillna(-999), index=True).values.tobytes()).hexdigest()[:16]

days = pd.date_range("2026-09-01", "2026-09-12", freq="D")
for off, NAT in ((1, N1), (2, N2)):
    X = pd.concat([build(NAT, off, d) for d in days])
    print(f"T+{off}: shape {X.shape} | hash {hsh(X)} | NaN {int(X.isna().sum().sum())} | "
          f"NaN-kolon {int((X.isna().sum()>0).sum())}")
