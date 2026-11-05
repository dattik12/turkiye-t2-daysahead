import glob, os, json, re
import pandas as pd
MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]
man = {}
for kind, pat in (("wind", r"C:\projeler\multinwp\wind\*.parquet"), ("solar", r"C:\projeler\multinwp\solar\*.parquet")):
    rows = {}
    for f in sorted(glob.glob(pat)):
        m = re.match(r"(.+)_(20\d\dQ\d)\.parquet", os.path.basename(f))
        if not m:
            continue
        prov, tag = m.group(1), m.group(2)
        d = pd.read_parquet(f)
        for mo in MODELS:
            cols = [c for c in d.columns if c.endswith(mo)]
            if not cols:
                continue
            n = int(d[cols[0]].notna().sum())
            tot = int(d[cols[0]].shape[0])
            key = tag
            rows.setdefault(key, {}).setdefault(mo, [0, 0])
            rows[key][mo][0] += n
            rows[key][mo][1] += tot
    man[kind] = {k: {mo: "%.0f%%" % (100*v[0]/v[1]) if v[1] else "0%" for mo, v in sorted(vv.items())} for k, vv in sorted(rows.items())}
out = r"C:\projeler\multinwp\COVERAGE_MANIFEST.json"
json.dump(man, open(out, "w"), indent=1)
print("manifest:", out)
for kind in man:
    print("\n==", kind, "==")
    for tag, mm in man[kind].items():
        print("  ", tag, " ".join("%s=%s" % (k.split("_")[0], v) for k, v in mm.items()))
