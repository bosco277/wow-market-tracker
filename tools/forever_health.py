"""Forever health check: is the Forever data complete, how is it stored, and are the auction houses separate markets?

    python tools/forever_health.py                       # everything, last 48 hours for the price comparison
    python tools/forever_health.py --compare-hours 24    # a shorter price comparison (fewer files to download)
    python tools/forever_health.py --compare-hours 0     # skip the price comparison

Reads the separate private Forever dataset (written by fetch_forever.py) and writes ONE plain-text report, out/forever_health.txt (a .txt on purpose: lines starting
with "=" turn into #ERROR! when a spreadsheet opens them). It only reads; nothing is written to any dataset.

What it checks
  1 WHAT IS COLLECTED   the sources file: region, rulesets, auction houses, regular and commodity files, and what the ruleset check found.
  2 EVERY HOUR?         from the file names alone (no downloads): the run times, gaps of more than 90 minutes, complete US Eastern days, freshness.
  3 SIZE AND GROWTH     MB and files per day, and when the 100 GB / 100,000 file guidelines of the free Hugging Face dataset would be reached.
  4 HOW IT IS STORED    the newest file of every source: its columns against what the retail files have, null rates of buyout / bid / unit_price, bid-only share,
                        and the real test: does the engine's own reader (normalize_frame, snapshots_from_blizzard_v2) take the file, and what kind does it call it?
  5 SEPARATE MARKETS?   the commodity files of the last hours side by side: shared auction ids, shared items, and how the same item is priced in each house.
  6 SNAPSHOT TIMING     Blizzard's Last-Modified against our fetch time, and snapshots stored twice.
Environment: HF_TOKEN (read access to the Forever dataset); FOREVER_HF_REPO to name another dataset. The dataset's name is blanked out of everything printed,
so no account name reaches a job log.
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine.loader import normalize_frame  # noqa: E402
from engine.schema import snapshots_from_blizzard_v2  # noqa: E402

HF_REPO = os.environ.get("FOREVER_HF_REPO", "bosco277/wow-forever-data-lake")
MARKERS_KEY = "system/forever_markers.json"
SOURCES_KEY = "system/forever_sources.json"
KEY = re.compile(r"^(?P<source>[^/]+)/year=(?P<y>\d{4})/month=(?P<m>\d{2})/day=(?P<d>\d{2})/hour_(?P<h>\d{2})_(?P<mi>\d{2})\.parquet$")
ET = "America/New_York"
GAP_HOURS = 1.5                      # two files further apart than this mean at least one hourly refresh is missing
HF_FREE_GB = 100.0                   # free Hugging Face storage (docs, section 2u)
HF_FILE_GUIDELINE = 100_000          # Hugging Face's guideline for files per repository
PVP_SET = (110, 113, 116)            # the PvP market set found on Nov 13 (Alliance, Horde, Blackwater); 113 and 116 come from the community post, not yet from a listing
MIN_UNITS = 20                       # price comparison: only items with at least this many units listed (median over the hours) in BOTH houses
COMMON = ["id", "item", "quantity", "time_left", "item_id", "variant_key", "source", "snapshot_ts", "scan_timestamp", "schema_version"]
REQUIRED = {"auctions": COMMON + ["buyout"], "commodities": COMMON + ["unit_price"]}
OPTIONAL = {"auctions": ["bid", "unit_price"], "commodities": []}
PRICE_COLS = ["buyout", "bid", "unit_price"]
COMPARE_COLS = ["id", "item_id", "unit_price", "quantity", "snapshot_ts", "scan_timestamp"]


def scrub(text):
    """Error text for logs and the report: the dataset name (and so the account name in it) is blanked out."""
    t = str(text)
    for part in (HF_REPO, HF_REPO.split("/")[0]):                       # the whole name first: replacing only the owner would leave "<forever dataset>/wow-forever-data-lake"
        if part:
            t = t.replace(part, "<forever dataset>")
    return t[:300]


# ------------------------------------------------------------------------------------------------- inventory (file names only)

def parse_key(key):
    """(source, fetch time as naive UTC) for a snapshot file path, or None for anything else (system files, folders)."""
    m = KEY.match(key)
    if not m:
        return None
    g = {k: int(v) if k != "source" else v for k, v in m.groupdict().items()}
    return g["source"], pd.Timestamp(g["y"], g["m"], g["d"], g["h"], g["mi"])


def kind_of(source):
    return "commodities" if str(source).endswith("_commodities") else "auctions"


def house_of(source):
    m = re.search(r"_h(\d+)", str(source))
    return int(m.group(1)) if m else None


def inventory(entries):
    """DataFrame [source, ts, size, key] (one row per snapshot file, oldest first) from [(path, bytes)]; also returns the entries that are not snapshot files."""
    rows, other = [], []
    for path, size in entries:
        p = parse_key(path)
        if p is None:
            other.append((path, size))
        else:
            rows.append((p[0], p[1], int(size or 0), path))
    inv = pd.DataFrame(rows, columns=["source", "ts", "size", "key"])
    if len(inv):
        inv = inv.sort_values(["ts", "source"]).reset_index(drop=True)
    return inv, other


def gaps_in(times, gap_hours=GAP_HOURS):
    """[(last file before the gap, first file after it, hours between)] for consecutive times further apart than `gap_hours`."""
    t = sorted(pd.Timestamp(x) for x in set(times))
    out = []
    for a, b in zip(t[:-1], t[1:]):
        h = (b - a).total_seconds() / 3600.0
        if h > gap_hours:
            out.append((a, b, h))
    return out


def missed_hours(gaps):
    return int(sum(max(int(round(h)) - 1, 1) for _, _, h in gaps))


def coverage(inv, now):
    """One row per source: first / last file, files, MB, missed hourly refreshes (from gaps of more than 90 minutes), coverage share, longest gap, hours since the newest file."""
    cols = ["source", "first", "last", "files", "mb", "missed", "coverage", "longest_gap_h", "age_h"]
    rows = []
    for source, d in inv.groupby("source"):
        g = gaps_in(d["ts"])
        files = len(d)
        missed = missed_hours(g)
        rows.append((source, d["ts"].min(), d["ts"].max(), files, d["size"].sum() / 1e6, missed, files / (files + missed), max([h for _, _, h in g], default=0.0),
                     (pd.Timestamp(now) - d["ts"].max()).total_seconds() / 3600.0))
    return pd.DataFrame(rows, columns=cols)


def day_table(times, now, gap_hours=GAP_HOURS):
    """Per US Eastern day: runs, longest gap touching the day, and whether the day is complete (a file at least every 90 minutes from before midnight to after the next one).
    Days the data starts or ends inside are marked, not counted."""
    t = sorted(pd.Timestamp(x) for x in set(times))
    cols = ["day", "runs", "longest_gap_h", "complete", "note"]
    if not t:
        return pd.DataFrame(columns=cols)
    et = [pd.Timestamp(x).tz_localize("UTC").tz_convert(ET) for x in t]
    first_day, last_day = et[0].normalize(), et[-1].normalize()
    arr = pd.DatetimeIndex(t).to_numpy()
    rows = []
    for day in pd.date_range(first_day.tz_localize(None), last_day.tz_localize(None), freq="D"):
        d0 = day.tz_localize(ET).tz_convert("UTC").tz_localize(None)
        d1 = (day + pd.Timedelta(days=1)).tz_localize(ET).tz_convert("UTC").tz_localize(None)
        lo, hi = int(np.searchsorted(arr, d0.to_datetime64())), int(np.searchsorted(arr, d1.to_datetime64()))
        inside = t[lo:hi]
        seq = ([t[lo - 1]] if lo > 0 else []) + inside + ([t[hi]] if hi < len(t) else [])
        diffs = [(b - a).total_seconds() / 3600.0 for a, b in zip(seq[:-1], seq[1:])]
        longest = max(diffs, default=0.0)
        note = ""
        if lo == 0:
            note = "data starts during this day"
        elif hi >= len(t):
            note = "day not over yet" if pd.Timestamp(now) < d1 else "data ends during this day"
        rows.append((day.date(), len(inside), longest, (not note) and longest <= gap_hours, note))
    return pd.DataFrame(rows, columns=cols)


def growth(inv, now, recent_days=7):
    """Totals and the recent daily rate: files and MB per day over the last `recent_days` days, overall and for the PvP set, and the days left to the two storage guidelines."""
    if inv.empty:
        return {}
    now = pd.Timestamp(now)
    recent = inv[inv["ts"] >= now - pd.Timedelta(days=recent_days)]
    span_days = max((recent["ts"].max() - recent["ts"].min()).total_seconds() / 86400.0, 1.0) if len(recent) > 1 else 1.0
    def block(d_all, d_recent):
        files_day, mb_day = len(d_recent) / span_days, d_recent["size"].sum() / 1e6 / span_days
        tot_files, tot_mb = len(d_all), d_all["size"].sum() / 1e6
        return dict(total_files=tot_files, total_mb=tot_mb, files_per_day=files_day, mb_per_day=mb_day,
                    days_to_files=(HF_FILE_GUIDELINE - tot_files) / files_day if files_day > 0 else np.inf,
                    days_to_storage=(HF_FREE_GB * 1000.0 - tot_mb) / mb_day if mb_day > 0 else np.inf)
    houses = inv["source"].map(house_of)
    pvp, pvp_recent = inv[houses.isin(PVP_SET)], recent[recent["source"].map(house_of).isin(PVP_SET)]
    per_source = (recent.groupby("source").agg(files=("ts", "size"), mb=("size", lambda s: s.sum() / 1e6)).assign(files_day=lambda d: d["files"] / span_days, mb_day=lambda d: d["mb"] / span_days)
                  .sort_values("mb_day", ascending=False))
    return dict(all=block(inv, recent), pvp=block(pvp, pvp_recent), per_source=per_source, span_days=span_days)


# ------------------------------------------------------------------------------------------------- what is collected

def sources_table(sources, latest_rows):
    """One row per (ruleset, house): name, which kinds are collected, and the listing counts of the newest files (regular, commodity)."""
    if not sources:
        return pd.DataFrame(columns=["region", "ruleset", "house", "name", "regular", "commodity", "listings_regular", "listings_commodity"])
    rows = {}
    for src, info in sources.items():
        k = (info.get("region"), info.get("ruleset"), info.get("house_id"))
        r = rows.setdefault(k, dict(region=k[0], ruleset=k[1], house=k[2], name=info.get("house_name"), regular=False, commodity=False, listings_regular=np.nan, listings_commodity=np.nan))
        if info.get("kind") == "commodities":
            r["commodity"], r["listings_commodity"] = True, latest_rows.get(src, np.nan)
        else:
            r["regular"], r["listings_regular"] = True, latest_rows.get(src, np.nan)
    return pd.DataFrame(list(rows.values())).sort_values(["region", "ruleset", "house"]).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------- how it is stored

def schema_report(df, source):
    """Columns against what the retail files have, null rates, bid-only share, and what the engine's own reader makes of the file."""
    kind = kind_of(source)
    cols = list(df.columns)
    out = dict(source=source, kind=kind, rows=int(len(df)), columns=cols, dtypes={c: str(df[c].dtype) for c in cols})
    out["missing"] = [c for c in REQUIRED[kind] if c not in cols]
    out["extra"] = [c for c in cols if c not in REQUIRED[kind] and c not in OPTIONAL[kind]]
    out["items"] = int(df["item_id"].nunique()) if "item_id" in cols else 0
    qty = pd.to_numeric(df["quantity"], errors="coerce") if "quantity" in cols else pd.Series(dtype=float)
    out["units"] = float(qty.sum()) if len(qty) else 0.0
    out["qty_gt1"] = float((qty > 1).mean()) if len(qty) else np.nan
    out["null"] = {c: float(df[c].isna().mean()) for c in PRICE_COLS if c in cols}
    if "buyout" in cols and "bid" in cols:
        out["bid_only"] = float((df["buyout"].isna() & df["bid"].notna()).mean())
    elif "buyout" in cols:
        out["bid_only"] = float(df["buyout"].isna().mean())
    else:
        out["bid_only"] = np.nan
    out["time_left"] = df["time_left"].value_counts().to_dict() if "time_left" in cols else {}
    out["variant_share"] = float(df["variant_key"].astype(str).str.contains(":", regex=False).mean()) if "variant_key" in cols and len(df) else np.nan
    out["dup_ids"] = int(df["id"].duplicated().sum()) if "id" in cols else 0
    out["schema_versions"] = sorted({int(x) for x in df["schema_version"].dropna().unique()}) if "schema_version" in cols else []
    if {"snapshot_ts", "scan_timestamp"} <= set(cols) and len(df):
        out["lag_min"] = float((pd.to_datetime(df["scan_timestamp"]).iloc[0] - pd.to_datetime(df["snapshot_ts"]).iloc[0]).total_seconds() / 60.0) if df["snapshot_ts"].notna().any() else np.nan
    else:
        out["lag_min"] = np.nan
    try:
        snaps = snapshots_from_blizzard_v2(normalize_frame(df, source))
        out["engine_ok"], out["engine_error"] = True, ""
        out["engine_kind"] = snaps[0].kind if snaps else ""
        out["engine_rows"] = int(sum(len(s.book) for s in snaps))
        out["engine_dropped"] = int(sum(s.meta.get("dropped_unpriced", 0) for s in snaps))
    except Exception as e:  # the point of the test is to find out what breaks
        out["engine_ok"], out["engine_error"] = False, scrub(f"{type(e).__name__}: {e}")
        out["engine_kind"], out["engine_rows"], out["engine_dropped"] = "", 0, 0
    out["kind_ok"] = (not out["engine_ok"]) or (out["engine_kind"] == ("commodity" if kind == "commodities" else "auction"))
    return out


# ------------------------------------------------------------------------------------------------- separate markets?

def floors_units(df):
    """Per item: the cheapest unit price (gold) and the units listed, from one commodity snapshot."""
    d = df[df["unit_price"].notna() & (df["unit_price"] > 0)]
    g = d.groupby("item_id").agg(floor=("unit_price", "min"), units=("quantity", "sum"))
    return g["floor"] / 10000.0, g["units"]


def panels(frames):
    """{source: [(file time, frame)]} -> {source: (floors, units)}, each a DataFrame with the file times as rows and item ids as columns."""
    out = {}
    for source, lst in frames.items():
        f, u = {}, {}
        for ts, df in lst:
            f[ts], u[ts] = floors_units(df)
        out[source] = (pd.DataFrame(f).T.sort_index(), pd.DataFrame(u).T.sort_index())
    return out


def id_overlap(latest_ids):
    """For each pair of sources: the share of the smaller one's auction ids that the other also has (1.0 = one is a copy of the other)."""
    rows = []
    names = sorted(latest_ids)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa, sb = latest_ids[a], latest_ids[b]
            smaller = min(len(sa), len(sb))
            rows.append((a, b, len(sa), len(sb), len(sa & sb), (len(sa & sb) / smaller) if smaller else np.nan))
    return pd.DataFrame(rows, columns=["a", "b", "ids_a", "ids_b", "shared", "share_of_smaller"])


def price_compare(pan, min_units=MIN_UNITS, min_hours=10):
    """For each pair of commodity houses, over the items listed with at least `min_units` units in both: the median price ratio (B / A) with its middle half, the share of items whose floors
    are identical in nearly every hour, and how closely the two floors move together (correlation of the hourly log floors, and of their hourly changes). Items are only compared on hours both have."""
    rows = []
    names = sorted(pan)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            (fa, ua), (fb, ub) = pan[a], pan[b]
            idx = fa.index.intersection(fb.index)
            fa, ua, fb, ub = fa.loc[idx], ua.loc[idx], fb.loc[idx], ub.loc[idx]
            ratios, same, lvl, chg, used = [], 0, [], [], 0
            for c in [c for c in fa.columns if c in fb.columns]:
                ok = (fa[c].notna() & fb[c].notna()).to_numpy()
                if ok.sum() < min_hours:
                    continue
                if np.nanmedian(ua[c].to_numpy()[ok]) < min_units or np.nanmedian(ub[c].to_numpy()[ok]) < min_units:
                    continue
                x, y = fa[c].to_numpy(float)[ok], fb[c].to_numpy(float)[ok]
                used += 1
                ratios.append(float(np.median(y) / np.median(x)))
                same += int(np.isclose(x, y, rtol=1e-6).mean() >= 0.95)
                lx, ly = np.log(x), np.log(y)
                if np.std(lx) > 0 and np.std(ly) > 0:
                    lvl.append(float(np.corrcoef(lx, ly)[0, 1]))
                dx, dy = np.diff(lx), np.diff(ly)
                if len(dx) >= min_hours and np.std(dx) > 0 and np.std(dy) > 0:
                    chg.append(float(np.corrcoef(dx, dy)[0, 1]))
            rows.append(dict(a=a, b=b, items=used, ratio_med=float(np.median(ratios)) if ratios else np.nan, ratio_lo=float(np.percentile(ratios, 25)) if ratios else np.nan,
                             ratio_hi=float(np.percentile(ratios, 75)) if ratios else np.nan, identical=(same / used) if used else np.nan,
                             corr_level=float(np.median(lvl)) if lvl else np.nan, corr_change=float(np.median(chg)) if chg else np.nan))
    return pd.DataFrame(rows)


def item_prices(pan, top=12):
    """The `top` items with the most units in the busiest commodity house: their median floor (gold) and units in every house, for reading the markets side by side."""
    if not pan:
        return pd.DataFrame()
    busiest = max(pan, key=lambda s: pan[s][1].median().sum() if pan[s][1].shape[1] else 0)
    units = pan[busiest][1].median().sort_values(ascending=False).head(top)
    rows = []
    for item in units.index:
        r = {"item_id": int(item)}
        for s, (f, u) in pan.items():
            r[f"{s}|price"] = float(f[item].median()) if item in f.columns else np.nan
            r[f"{s}|units"] = float(u[item].median()) if item in u.columns else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def timing_stats(frames):
    """Per commodity source over the downloaded hours: files, distinct Last-Modified times (fewer than files = a snapshot stored twice), and the median / worst lag between Blizzard's
    Last-Modified and our fetch, in minutes."""
    rows = []
    for source, lst in frames.items():
        snaps, lags = [], []
        for ts, df in lst:
            if "snapshot_ts" in df.columns and df["snapshot_ts"].notna().any():
                s = pd.to_datetime(df["snapshot_ts"]).iloc[0]
                snaps.append(s)
                lags.append((pd.Timestamp(ts) - s).total_seconds() / 60.0)
        rows.append(dict(source=source, files=len(lst), distinct=len(set(snaps)), lag_med=float(np.median(lags)) if lags else np.nan, lag_max=float(np.max(lags)) if lags else np.nan))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- report

def _pct(x, d=0):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def _n(x, f="{:,.0f}"):
    return "n/a" if x is None or (isinstance(x, float) and (np.isnan(x))) else f.format(x)


def _when(ts):
    return "n/a" if ts is None or pd.isna(ts) else pd.Timestamp(ts).strftime("%a %b %d %H:%M")


def readout(r):
    """The short plain-words verdict at the top: (level, text) lines, level one of OK / WATCH / PROBLEM."""
    out = []
    inv = r["inv"]
    if inv.empty:
        return [("PROBLEM", "No snapshot files found in the Forever dataset.")]
    cov, days, gr = r["coverage"], r["days"], r["growth"]
    union = r["union"]
    span_days = (inv["ts"].max() - inv["ts"].min()).total_seconds() / 86400.0
    n_complete = int(days["complete"].sum()) if len(days) else 0
    cov_all = union["coverage"]
    lvl = "OK" if cov_all >= 0.97 else ("WATCH" if cov_all >= 0.90 else "PROBLEM")
    out.append((lvl, f"Collected {_when(inv['ts'].min())} to {_when(inv['ts'].max())} UTC ({span_days:.1f} days): {union['runs']:,} runs, about {union['missed']} hourly refreshes missing "
                     f"({_pct(cov_all, 1)} present); longest gap {union['longest_gap_h']:.1f} h; {n_complete} complete US Eastern days."))
    age = (pd.Timestamp(r["now"]) - inv["ts"].max()).total_seconds() / 3600.0
    out.append(("OK" if age <= 2.5 else ("WATCH" if age <= 6 else "PROBLEM"), f"Newest file is {age:.1f} h old" + ("." if age <= 2.5 else ": the collector may have stopped (Actions > Forever Auction Extractor).")))
    lag = cov[cov["coverage"] < cov_all - 0.005]
    out.append(("OK" if lag.empty else "WATCH", "Every source has the same hours as the whole run." if lag.empty else f"{len(lag)} source(s) have fewer hours than the whole run (listed in section 2)."))
    if r["schema"]:
        bad = [s for s in r["schema"] if not s["engine_ok"]]
        wrong = [s for s in r["schema"] if not s["kind_ok"]]
        miss = [s for s in r["schema"] if s["missing"]]
        out.append(("OK" if not (bad or miss) else "PROBLEM", f"The engine's own reader takes {len(r['schema']) - len(bad)} of {len(r['schema'])} newest files" + ("." if not bad else f"; {len(bad)} fail (section 4).")))
        com = [s for s in r["schema"] if s["kind"] == "commodities"]
        if com:
            out.append(("WATCH" if wrong else "OK", f"{len(wrong)} of {len(com)} commodity sources would be called 'auction' by the engine (it only knows the name 'commodities'); fix before the lift and shift." if wrong
                        else "Commodity sources are classified as commodities."))
    if r["overlap"] is not None and len(r["overlap"]):
        dup = r["overlap"][r["overlap"]["share_of_smaller"] >= 0.9]
        out.append(("WATCH" if len(dup) else "OK", f"{len(dup)} pair(s) of commodity houses share 90% or more of their auction ids (copies, not separate markets)." if len(dup)
                    else "No two commodity houses share auction ids: they are separate markets."))
    g = gr.get("all") if gr else None
    if g:
        lvl = "OK" if g["days_to_files"] > 180 else ("WATCH" if g["days_to_files"] > 60 else "PROBLEM")
        out.append((lvl, f"Growth: {g['mb_per_day']:,.0f} MB and {g['files_per_day']:,.0f} files a day; the 100,000-file guideline in about {_n(g['days_to_files'])} days, 100 GB in about {_n(g['days_to_storage'])} days."))
    return out


def format_report(r):
    L = [f"### FOREVER HEALTH CHECK (built {_when(r['now'])} UTC)", ""]
    for lvl, text in readout(r):
        L.append(f"  [{lvl:7s}] {text}")
    inv = r["inv"]
    if inv.empty:
        return "\n".join(L)

    L += ["", "### 1. WHAT IS BEING COLLECTED"]
    st = r["sources_table"]
    if st.empty:
        L.append("  No sources file found (system/forever_sources.json); the table below comes from the file names only.")
    else:
        regions, rulesets = sorted({str(x) for x in st["region"].dropna()}), sorted({str(x) for x in st["ruleset"].dropna()})
        L.append(f"  Regions in the dataset: {', '.join(regions) or 'n/a'}. Rulesets read: {', '.join(rulesets) or 'n/a'}. Auction houses: {len(st)} ({int(st['regular'].sum())} with regular auctions, {int(st['commodity'].sum())} with commodities).")
        L.append("  Forever has no realms: a market is an auction house. Counts are the listings in each house's newest file.")
        L.append(f"  {'region':6s} {'rs':>3s} {'house':>6s} {'name':26s} {'regular listings':>17s} {'commodity listings':>19s} {'MB/day':>8s}")
        per = r["growth"].get("per_source", pd.DataFrame()) if r["growth"] else pd.DataFrame()
        for _, x in st.iterrows():
            mb = sum(float(per["mb_day"].get(s, 0.0)) for s in per.index if house_of(s) == x["house"]) if len(per) else np.nan
            L.append(f"  {str(x['region']):6s} {_n(x['ruleset']):>3s} {_n(x['house']):>6s} {str(x['name'])[:26]:26s} {_n(x['listings_regular']):>17s} {_n(x['listings_commodity']):>19s} {_n(mb, '{:.1f}'):>8s}")
    if r["check"]:
        L.append(f"  Ruleset check (stored by the fetcher): {scrub(json.dumps(r['check'], sort_keys=True))}")
    else:
        L.append("  Ruleset check: none stored in the markers file.")

    L += ["", "### 2. DO WE HAVE EVERY HOUR?"]
    u = r["union"]
    L.append(f"  Whole runs (all sources are written together, so one run is one time): {u['runs']:,} runs from {_when(u['first'])} to {_when(u['last'])} UTC. Gaps of more than 90 minutes: {u['n_gaps']}, "
             f"about {u['missed']} hourly refreshes missing, longest {u['longest_gap_h']:.1f} h.")
    for a, b, h in u["gaps"][:25]:
        L.append(f"    gap {h:5.1f} h: {_when(a)} to {_when(b)} UTC")
    if len(u["gaps"]) > 25:
        L.append(f"    ... {len(u['gaps']) - 25} more")
    L.append("  Per US Eastern day (complete = a file at least every 90 minutes from before midnight to after the next one):")
    L.append(f"    {'day':12s} {'runs':>5s} {'longest gap h':>14s}  status")
    for _, d in r["days"].iterrows():
        L.append(f"    {str(d['day']):12s} {int(d['runs']):5d} {d['longest_gap_h']:14.1f}  {'complete' if d['complete'] else (d['note'] or 'GAP')}")
    cov = r["coverage"]
    odd = cov[(cov["coverage"] < u["coverage"] - 0.005) | (cov["age_h"] > 3.0)]
    if odd.empty:
        L.append(f"  All {len(cov)} sources have the same hours as the whole run and are fresh.")
    else:
        L.append(f"  Sources that differ from the whole run ({len(odd)} of {len(cov)}):")
        for _, x in odd.head(25).iterrows():
            L.append(f"    {x['source']:42s} files {int(x['files']):5d} missed {int(x['missed']):4d} coverage {_pct(x['coverage'], 1)} longest gap {x['longest_gap_h']:.1f} h newest {x['age_h']:.1f} h old")
        if len(odd) > 25:
            L.append(f"    ... {len(odd) - 25} more")
        L.append("  (A house that is empty in some hours is kept out of those hours by the fetcher, so it shows gaps that are not failures; a house that is empty most of the time is not worth keeping.)")

    L += ["", "### 3. SIZE AND GROWTH"]
    gr = r["growth"]
    a, p = gr["all"], gr["pvp"]
    L.append(f"  Stored now: {a['total_files']:,} snapshot files, {a['total_mb'] / 1000:.2f} GB.")
    L.append(f"  Recent rate (last {gr['span_days']:.1f} days), all sources: {a['files_per_day']:,.0f} files and {a['mb_per_day']:,.0f} MB a day. At that rate the {HF_FILE_GUIDELINE:,}-file guideline "
             f"arrives in about {_n(a['days_to_files'])} days and the {HF_FREE_GB:.0f} GB limit in about {_n(a['days_to_storage'])} days (compaction, which is not built, would remove the file limit).")
    if p["total_files"]:
        L.append(f"  Only the PvP set (houses {', '.join(map(str, PVP_SET))}): {p['files_per_day']:,.0f} files and {p['mb_per_day']:,.0f} MB a day; the file guideline in about {_n(p['days_to_files'])} days, "
                 f"the storage limit in about {_n(p['days_to_storage'])} days.")
    L.append(f"  {'source':44s} {'files/day':>10s} {'MB/day':>8s}")
    for s, x in gr["per_source"].head(30).iterrows():
        L.append(f"  {s:44s} {x['files_day']:10.1f} {x['mb_day']:8.2f}")
    if len(gr["per_source"]) > 30:
        L.append(f"  ... {len(gr['per_source']) - 30} smaller sources not shown")
    if r["other_files"]:
        L.append(f"  Other files in the dataset (not snapshots): {len(r['other_files'])}, {sum(s for _, s in r['other_files']) / 1e6:.1f} MB.")

    L += ["", "### 4. HOW THE DATA IS STORED"]
    L.append("  Retail's files and Forever's are written by the same frame code, so the expected columns are the same. Regular auctions carry buyout (a TOTAL for the stack) and sometimes bid;")
    L.append("  commodities carry unit_price (per unit). The engine reads either one. 'Engine kind' is what the engine's own reader (snapshots_from_blizzard_v2) calls the source.")
    sch = r["schema"]
    if not sch:
        L.append("  No files could be read.")
    else:
        L.append(f"  {'source':42s} {'kind':11s} {'rows':>7s} {'items':>6s} {'units':>9s} {'qty>1':>6s} {'no buyout':>10s} {'unpriced':>9s} {'engine kind':>12s}  columns")
        for s in sch:
            unpriced = s["engine_dropped"] / max(s["engine_dropped"] + s["engine_rows"], 1) if s["engine_ok"] else np.nan
            cols = "ok" if not (s["missing"] or s["extra"]) else (("MISSING " + ",".join(s["missing"]) + " ") if s["missing"] else "") + (("EXTRA " + ",".join(s["extra"])) if s["extra"] else "")
            eng = (s["engine_kind"] + ("" if s["kind_ok"] else " (WRONG)")) if s["engine_ok"] else "FAILS"
            L.append(f"  {s['source'][:42]:42s} {s['kind']:11s} {s['rows']:7,d} {s['items']:6,d} {_n(s['units']):>9s} {_pct(s['qty_gt1']):>6s} {_pct(s['bid_only'], 1):>10s} {_pct(unpriced, 1):>9s} {eng:>12s}  {cols}")
        for s in sch:
            if not s["engine_ok"]:
                L.append(f"    {s['source']}: engine error: {s['engine_error']}")
        first = {k: next((s for s in sch if s["kind"] == k), None) for k in ("auctions", "commodities")}
        for k, s in first.items():
            if s:
                L.append(f"  Columns of the newest {k} file ({s['source']}): " + ", ".join(f"{c} {s['dtypes'][c]}" for c in s["columns"]))
        tl = {}
        for s in sch:
            for k, v in s["time_left"].items():
                tl[(s["kind"], k)] = tl.get((s["kind"], k), 0) + v
        for kind in ("auctions", "commodities"):
            tot = sum(v for (k, _), v in tl.items() if k == kind)
            if tot:
                L.append(f"  time_left values, {kind}: " + ", ".join(f"{k} {_pct(v / tot)}" for (kk, k), v in sorted(tl.items()) if kk == kind))
        vs = [s["variant_share"] for s in sch if s["kind"] == "auctions" and s["variant_share"] == s["variant_share"]]
        if vs:
            L.append(f"  Regular auctions with gear-style variants (variant_key has a hash): {_pct(float(np.mean(vs)))} of rows.")
        dups = sum(s["dup_ids"] for s in sch)
        L.append(f"  Repeated auction ids inside a file: {dups}.")
        com_wrong = [s for s in sch if not s["kind_ok"]]
        if com_wrong:
            L.append("  LIFT AND SHIFT: the engine marks a snapshot a commodity only when its source is named exactly 'commodities' (engine/schema.py). Forever's commodity sources are named like "
                     f"'{com_wrong[0]['source']}', so today they would be read as regular auctions. The adapter has to set the kind from the '_commodities' suffix.")

    L += ["", "### 5. ARE THE HOUSES SEPARATE MARKETS?"]
    ov = r["overlap"]
    L.append("  Auction ids shared between two houses, newest commodity files (1.00 = one is a copy of the other, 0.00 = no overlap):")
    if ov is None or ov.empty:
        L.append("    fewer than two commodity sources to compare.")
    else:
        for _, x in ov.iterrows():
            L.append(f"    {x['a']:40s} vs {x['b']:40s} ids {int(x['ids_a']):6,d} / {int(x['ids_b']):6,d}, shared {int(x['shared']):6,d} ({_pct(x['share_of_smaller'])} of the smaller)")
    pc = r["compare"]
    L.append(f"  Prices of the same items in different houses (commodities, {r['compare_note']}):")
    if pc is None or pc.empty:
        L.append("    nothing to compare (no price window downloaded, or fewer than two commodity sources).")
    else:
        L.append(f"    only items with at least {MIN_UNITS} units in both houses; ratio = house B price / house A price; 'identical' = items whose floors match in 95% of hours")
        L.append(f"    {'A':40s} {'B':40s} {'items':>5s} {'price ratio B/A (middle half)':>30s} {'identical':>9s} {'corr (level)':>13s} {'corr (hourly change)':>21s}")
        for _, x in pc.iterrows():
            rr = "n/a" if x["items"] == 0 else f"{x['ratio_med']:.2f} ({x['ratio_lo']:.2f} to {x['ratio_hi']:.2f})"
            L.append(f"    {x['a']:40s} {x['b']:40s} {int(x['items']):5d} {rr:>30s} {_pct(x['identical']):>9s} {_n(x['corr_level'], '{:+.2f}'):>13s} {_n(x['corr_change'], '{:+.2f}'):>21s}")
        L.append("    Reading it: a ratio near 1.00 with high correlation = the houses behave like one market; a ratio far from 1 or low correlation = separate markets that need their own tables.")
    ip = r["item_prices"]
    if ip is not None and len(ip):
        names = r["names"]
        srcs = [c.split("|")[0] for c in ip.columns if c.endswith("|price")]
        L.append(f"  The {len(ip)} items with the most units in the busiest house: median floor in gold (units) per house:")
        L.append("    " + f"{'item':34s}" + "".join(f"{('h' + str(house_of(s))):>22s}" for s in srcs))
        for _, x in ip.iterrows():
            nm = names.get(int(x["item_id"]), "")
            L.append("    " + f"{(str(int(x['item_id'])) + ' ' + nm)[:34]:34s}" + "".join(f"{_n(x[s + '|price'], '{:,.2f}') + ' (' + _n(x[s + '|units']) + ')':>22s}" for s in srcs))

    L += ["", "### 6. SNAPSHOT TIMING"]
    tm = r["timing"]
    if tm is None or tm.empty:
        L.append("  No price window downloaded.")
    else:
        L.append("  Lag = our fetch time minus Blizzard's Last-Modified, in minutes. 'distinct' below 'files' means the same snapshot was stored twice.")
        L.append(f"  {'source':42s} {'files':>6s} {'distinct':>9s} {'lag median':>11s} {'lag worst':>10s}")
        for _, x in tm.iterrows():
            L.append(f"  {x['source']:42s} {int(x['files']):6d} {int(x['distinct']):9d} {_n(x['lag_med'], '{:.0f}'):>11s} {_n(x['lag_max'], '{:.0f}'):>10s}")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------- orchestration

def load_names(ids, path=None):
    """{item id: name} for the given ids from items.json (a retail / Classic item list in the repository), or {} when it is not there."""
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "items.json")
    if not os.path.exists(path):
        return {}
    want = {int(i) for i in ids}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {int(d["itemId"]): d.get("name", "") for d in data if isinstance(d, dict) and int(d.get("itemId", -1)) in want}
    except Exception:
        return {}


def union_summary(inv):
    times = sorted(set(inv["ts"]))
    g = gaps_in(times)
    runs = len(times)
    missed = missed_hours(g)
    return dict(runs=runs, first=times[0], last=times[-1], gaps=g, n_gaps=len(g), missed=missed, coverage=runs / (runs + missed), longest_gap_h=max([h for _, _, h in g], default=0.0))


def run(args, io, now=None, names=None):
    """io needs list_entries() -> [(path, bytes)], read_json(key) -> dict, read_frame(key, columns=None) -> DataFrame. Returns the results dict (the report is r['text'])."""
    now = pd.Timestamp(now) if now is not None else pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    entries = io.list_entries()
    inv, other = inventory(entries)
    r = dict(now=now, inv=inv, other_files=other)
    if inv.empty:
        r["text"] = format_report(dict(r, coverage=None))
        return r
    if getattr(args, "days", 0):
        inv = inv[inv["ts"] >= inv["ts"].max() - pd.Timedelta(days=args.days)].reset_index(drop=True)
        r["inv"] = inv
    r["union"] = union_summary(inv)
    r["coverage"] = coverage(inv, now)
    r["days"] = day_table(inv["ts"], now)
    r["growth"] = growth(inv, now)
    sources = io.read_json(SOURCES_KEY) or {}
    markers = io.read_json(MARKERS_KEY) or {}
    r["check"] = markers.get("_ruleset_check")

    newest = inv.sort_values("ts").groupby("source").tail(1).set_index("source")["key"].to_dict()
    schema, rows, ids = [], {}, {}
    for source in sorted(newest):
        try:
            df = io.read_frame(newest[source])
        except Exception as e:
            schema.append(dict(source=source, kind=kind_of(source), rows=0, columns=[], dtypes={}, missing=["unreadable"], extra=[], items=0, units=0.0, qty_gt1=np.nan, null={}, bid_only=np.nan,
                               time_left={}, variant_share=np.nan, dup_ids=0, schema_versions=[], lag_min=np.nan, engine_ok=False, engine_error=scrub(f"could not read the file: {e}"),
                               engine_kind="", engine_rows=0, engine_dropped=0, kind_ok=True))
            continue
        schema.append(schema_report(df, source))
        rows[source] = len(df)
        if kind_of(source) == "commodities" and "id" in df.columns:
            ids[source] = set(df["id"].dropna().astype("int64"))
    r["schema"] = schema
    r["sources_table"] = sources_table(sources, rows)
    r["overlap"] = id_overlap(ids) if len(ids) >= 2 else pd.DataFrame()

    r["compare"], r["item_prices"], r["timing"], r["compare_note"] = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "no window downloaded"
    hours = getattr(args, "compare_hours", 0)
    com_sources = sorted(s for s in inv["source"].unique() if kind_of(s) == "commodities")
    if hours and com_sources:
        last = inv["ts"].max()
        win = inv[inv["source"].isin(com_sources) & (inv["ts"] > last - pd.Timedelta(hours=hours))]
        cap = getattr(args, "max_mb", 0) * 1e6
        if cap and win["size"].sum() > cap:
            times = sorted(set(win["ts"]), reverse=True)
            keep, total = [], 0
            for t in times:
                sz = win.loc[win["ts"] == t, "size"].sum()
                if total + sz > cap:
                    break
                keep.append(t)
                total += sz
            win = win[win["ts"].isin(keep)]
        frames = {}
        for _, x in win.iterrows():
            try:
                frames.setdefault(x["source"], []).append((x["ts"], io.read_frame(x["key"], COMPARE_COLS)))
            except Exception as e:
                frames.setdefault(x["source"], [])
                print(f"could not read one file of {x['source']}: {scrub(e)}", flush=True)
        n_runs = len({t for lst in frames.values() for t, _ in lst})
        r["compare_note"] = f"{n_runs} hourly runs, {_when(win['ts'].min())} to {_when(win['ts'].max())} UTC, {win['size'].sum() / 1e6:,.0f} MB read" if len(win) else "no files in the window"
        pan = panels({s: sorted(l, key=lambda z: z[0]) for s, l in frames.items() if l})
        pan = {s: v for s, v in pan.items() if v[0].shape[1]}
        r["compare"] = price_compare(pan) if len(pan) >= 2 else pd.DataFrame()
        r["item_prices"] = item_prices(pan)
        r["timing"] = timing_stats({s: l for s, l in frames.items() if l})
    ids_for_names = set(int(i) for i in r["item_prices"]["item_id"]) if len(r["item_prices"]) else set()
    r["names"] = names if names is not None else load_names(ids_for_names)
    r["text"] = format_report(r)
    return r


class HFIO:
    """Reads the Forever dataset on Hugging Face (needs huggingface_hub and HF_TOKEN)."""

    def __init__(self, repo=HF_REPO, token=None):
        from huggingface_hub import HfApi
        self.repo, self.token = repo, token or os.environ.get("HF_TOKEN")
        if not self.token:
            raise SystemExit("forever_health needs HF_TOKEN (read access to the Forever dataset).")
        self.api = HfApi(token=self.token)

    def _retry(self, fn, what, attempts=4, delay=5.0):
        import time
        for i in range(attempts):
            try:
                return fn()
            except Exception as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                transient = code is None or code >= 500 or code == 429
                if i == attempts - 1 or not transient:
                    raise RuntimeError(f"{what}: {scrub(e)}") from None
                time.sleep(delay * (2 ** i))

    def list_entries(self):
        def go():
            return [(e.path, getattr(e, "size", 0) or 0) for e in self.api.list_repo_tree(self.repo, repo_type="dataset", recursive=True) if hasattr(e, "size")]
        return self._retry(go, "list the dataset")

    def _local(self, key):
        from huggingface_hub import hf_hub_download
        return self._retry(lambda: hf_hub_download(self.repo, key, repo_type="dataset", token=self.token), f"read {key.split('/')[0]}")

    def read_json(self, key):
        try:
            with open(self._local(key), encoding="utf-8") as f:
                return json.load(f)
        except RuntimeError as e:
            if "404" in str(e) or "not found" in str(e).lower():
                return {}
            raise

    def read_frame(self, key, columns=None):
        return pd.read_parquet(self._local(key), columns=columns)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=0, help="only look at the last N days of files (0 = everything)")
    ap.add_argument("--compare-hours", type=int, default=48, help="hours of commodity files to download for the price comparison (0 = skip)")
    ap.add_argument("--max-mb", type=int, default=800, help="never download more than this many MB for the price comparison (the window is shortened to fit)")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)
    r = run(args, HFIO())
    print(r["text"])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "forever_health.txt"), "w", encoding="utf-8") as f:
        f.write(r["text"] + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
