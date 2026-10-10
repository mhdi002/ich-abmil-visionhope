"""Exclusion list: orientation-uncertain scans, manually reviewed non-axial reformats, byte-identical duplicates.

Manual review results live in outputs/orientation/manual_exclusions.csv (key,reason); scan keys contain
accession numbers and are kept out of version control.
"""
import pandas as pd

from utils.config import load_config, p

cfg = load_config()
d = p(cfg, "outputs") / "orientation"
manual_csv = d / "manual_exclusions.csv"
MANUAL = dict(pd.read_csv(manual_csv).itertuples(index=False)) if manual_csv.exists() else {}
o = pd.read_csv(d / "orientation.csv").set_index("key")
rows = [dict(key=k, axcodes=o.loc[k, "axcodes"], reason="orientation uncertain (conf < 0.9)")
        for k in o.index[o["orient_status"] == "uncertain"]]
rows += [dict(key=k, axcodes=o.loc[k, "axcodes"] if k in o.index else "", reason=v) for k, v in MANUAL.items()]
dup_csv = p(cfg, "outputs") / "leakage_audit" / "exact_duplicates.csv"
if dup_csv.exists():
    rows += [dict(key=r.key, axcodes=o.loc[r.key, "axcodes"] if r.key in o.index else "",
                  reason=f"byte-identical file to {r.duplicate_of}")
             for r in pd.read_csv(dup_csv).itertuples()]
ex = pd.DataFrame(rows).drop_duplicates("key")
ex.to_csv(d / "exclude.csv", index=False)
print(f"{len(ex)} scans excluded")
print(ex.to_string())
