"""Forever daily tables: one row per auction house, item and US Eastern day, built from the hourly commodity files and saved in the Forever dataset (WoW Forever only; retail is not touched).

    python tools/forever_daily.py                      # read the last 4 days of files, build the rows and REPORT; saves nothing
    python tools/forever_daily.py --save               # the same, then store the days that are new or changed (one commit)
    python tools/forever_daily.py --days 10 --end 2026-11-20 --save     # backfill: the 10 UTC days ending Nov 20 (run it in chunks; each run leaves its first and last 12 hours to its neighbours)
    python tools/forever_daily.py --houses 110,116     # only these houses (default: every house the fetcher stores, empty houses left out)
    python tools/forever_daily.py --no-sales           # skip the sales classifier: prices and depth only (faster); the sales columns are left blank
    python tools/forever_daily.py --save --rebuild     # new rows replace stored ones even where the stored row had more usable hours

What a row holds and how days are merged: see engine/foreverdaily.py. Stored as daily/d_YYYY-MM-DD.csv.gz (all houses in one file per day); a day is only written when its content changed.
Reads only the Forever dataset: no Blizzard calls, no secrets other than HF_TOKEN (write access to save, read access otherwise). FOREVER_HF_REPO names another dataset.
Writes ONE plain-text report, out/forever_daily.txt (a .txt on purpose). The dataset's name is blanked out of everything printed, so no account name reaches a job log.
"""
import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd  # noqa: E402

from engine import foreverdaily as FD  # noqa: E402
from engine import forever as FV  # noqa: E402
from engine import forevertables as T  # noqa: E402
from engine.loader import HFStore  # noqa: E402
import forever_tables as FT  # noqa: E402

MAX_DAYS = 60
LOOK_HOUSES = (110, 113, 116)          # which house the "look at the data" section prefers (the PvP set)


def et(ts):
    """A naive-UTC timestamp as 'MM-DD HH:MM' US Eastern."""
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t
    return t.tz_convert(FD.TZ).strftime("%m-%d %H:%M")


def parse_end(text, now):
    t = str(text or "").strip()
    if not t:
        return now.date()
    try:
        return date.fromisoformat(t)
    except ValueError:
        raise SystemExit(f"--end must be a date like 2026-11-25, not {t!r}")


def read_table(store, key, kind):
    data = FT.read_key(store, key)
    return T.from_csv_gz(data, kind) if data else T.tidy(None, kind)


def read_stored(store, days):
    """{key: bytes} for the stored day files of these days."""
    wanted = {FD.day_key(d) for d in days}
    out = {}
    for o in store.list(f"{FD.DAILY_DIR}/"):
        if o.key in wanted and o.size > 0:
            out[o.key] = store.read_range(o.key, 0, o.size)
    return out


def _fmt(v, nd=2):
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    return f"{v:,.{nd}f}"


def run(args, store, now=None, base_reader=None, commit=FT.hf_commit):
    now = now or datetime.now(timezone.utc)
    lines = []
    out = lines.append
    code = 0
    days = int(args.days)
    if not 1 <= days <= MAX_DAYS:
        return dict(text=f"--days must be between 1 and {MAX_DAYS}, not {days}.", code=1, files={}, changed=False, houses=[])
    end = parse_end(args.end, now)
    start = end - timedelta(days=days - 1)
    sales = not bool(getattr(args, "no_sales", False))
    rebuild = bool(getattr(args, "rebuild", False))
    save = bool(getattr(args, "save", False))

    avail = FV.houses_in_store(store)
    if not avail:
        return dict(text="### forever daily\nThe sources file could not be read, so no house is known. Wrong dataset name, or a token without access to it.", code=1, files={}, changed=False, houses=[])
    avail = [h for h in avail if h not in T.EMPTY_HOUSES]
    houses = FV.parse_houses(args.houses, avail)
    unknown = [h for h in houses if h not in avail]
    houses = [h for h in houses if h in avail]

    out("### forever daily")
    out(f"run at {now.strftime('%Y-%m-%d %H:%M')} UTC; files read: UTC days {start} to {end} ({days} day{'s' if days != 1 else ''}); houses: {', '.join(map(str, houses)) or 'none'}; "
        + ("SAVE" if save else "REPORT ONLY (save = no)") + ("; REBUILD" if rebuild else ""))
    out("Sales columns: " + ("ON. They are LOWER BOUNDS: a listing posted and sold inside one hour is in no snapshot, so use them to compare items and days, not as true volume."
                            if sales else "OFF (--no-sales): prices and depth are complete, the sales columns are left blank."))
    if unknown:
        out(f"Skipped: house(s) {', '.join(map(str, unknown))} are not in the dataset's sources file (or are empty houses).")
    if not houses:
        out("Nothing to do: no house to read.")
        return dict(text="\n".join(lines), code=1, files={}, changed=False, houses=[])

    # ------------------------------------------------------------------ 1 reading
    out("")
    out("### 1 reading the hourly commodity files (one house at a time, never pooled)")
    out(f"  {'house':>5} {'files':>6} {'snapshots':>9}  {'first (ET)':<12} {'last (ET)':<12} {'gaps>90m':>8} {'longest gap':>11} {'dup ids':>7} {'skipped':>7} {'rows built':>10}")
    built = []
    failed = []
    for h in houses:
        try:
            hd = FV.load_house(store, h, start, end, kinds=("commodity",), on_error="skip", base_reader=base_reader)
            sl = hd.commodity
            snaps = sl.snapshots
            gaps = FV.snapshot_gaps(snaps)
            rows = FD.build_house(snaps, h, sales=sales)
            first = et(snaps[0].ts) if snaps else "-"
            last = et(snaps[-1].ts) if snaps else "-"
            longest = f"{max(g[2] for g in gaps):.1f} h" if gaps else "-"
            out(f"  {h:>5} {sl.files:>6} {len(snaps):>9}  {first:<12} {last:<12} {len(gaps):>8} {longest:>11} {sl.dups.duplicate_rows:>7} {len(sl.result.skipped):>7} {len(rows):>10,}")
            for key, msg in sl.result.skipped[:3]:
                out(f"        skipped file {FV.scrub(key)}: {FV.scrub(msg)}")
            if sl.result.skipped:
                code = 1
            if len(rows):
                built.append(rows)
            snap_days = FD.snapshot_days(snaps)
            built_days = sorted(rows["day"].unique()) if len(rows) else []
            part = [d for d in built_days if snap_days.get(d, 0) < 20]
            if part:
                out("        days with fewer than 20 snapshots: " + ", ".join(f"{d} ({snap_days.get(d, 0)})" for d in part))
            del snaps, hd, sl
        except Exception as e:                                  # one house failing must not hide the others
            failed.append(h)
            code = 1
            out(f"  {h:>5} FAILED: {FV.scrub(e)}")
    if failed:
        out(f"  House(s) {', '.join(map(str, failed))} failed; the others are still stored below.")
    if not built:
        out("")
        out("No daily rows could be built (fewer than 3 snapshots, or no usable hours, for every house).")
        return dict(text="\n".join(lines), code=1, files={}, changed=False, houses=houses)
    new = pd.concat(built, ignore_index=True)

    # ------------------------------------------------------------------ 2 days
    out("")
    out("### 2 days built (US Eastern days; a complete row has 18+ usable hours; the first and last 12 hours of the files read are left out)")
    summ = FD.day_summary(new)
    out(f"  {'day':<11} {'houses':>6} {'item-day rows':>13} {'complete rows':>13}")
    for r in summ.itertuples():
        out(f"  {r.day:<11} {r.houses:>6} {r.rows:>13,} {r.complete_rows:>13,}")

    # ------------------------------------------------------------------ 3 storing
    out("")
    out("### 3 storing")
    existing = read_stored(store, summ["day"].tolist())
    files, plan = FD.plan_writes(new, existing, rebuild=rebuild)
    for p in plan:
        what = {"new": f"new ({p['rows']:,} rows)", "changed": f"changed ({p['rows_before']:,} -> {p['rows']:,} rows)", "unchanged": "unchanged"}[p["status"]]
        out(f"  {p['key']}: {FT.kb(p['bytes'])}, {what}")
    changed = bool(files)
    if not changed:
        out("Nothing to store: every day is already up to date.")
    elif save:
        commit(files, f"forever daily: {', '.join(sorted(FD.parse_day_key(k) for k in files))}")
        out(f"Saved {len(files)} file(s) to the Forever dataset in one commit.")
    else:
        out(f"Report only: {len(files)} file(s) would be stored. Run again with save = yes to store them.")

    # the rows as stored (merged with what was already there), for the two looks below
    merged = {p["day"]: FD.from_csv_gz(files.get(p["key"], existing.get(p["key"]))) for p in plan}
    pick = [d for d in sorted(merged) if int((merged[d]["hours"] >= FD.COMPLETE_HOURS).sum()) > 0]
    look_day = pick[-1] if pick else sorted(merged)[-1]
    look = merged[look_day]

    # ------------------------------------------------------------------ 4 a look at the data
    out("")
    house = next((h for h in LOOK_HOUSES if h in set(look["house"])), int(look["house"].iloc[0]))
    hr = look[look["house"] == house].sort_values("gold_day", ascending=False).head(8)
    items = read_table(store, T.KEYS["items"], "items")
    names = dict(zip(items["item_id"].astype(int), items["name"])) if len(items) else {}
    out(f"### 4 a look at the data: {look_day}, house {house}, the 8 items with the most gold in play (gold per unit)")
    out(f"  {'item':<34} {'price_day':>10} {'price_min':>10} {'buy20':>10} {'buy200':>10} {'units':>10} {'gold in play':>13} {'sold_low/day':>12} {'turnover':>8} {'active':>6} {'hours':>5}")
    for r in hr.itertuples():
        label = f"{str(names.get(int(r.item_id), '?'))[:26]} ({r.item_id})"
        turn = "-" if r.turnover_day != r.turnover_day else f"{100 * r.turnover_day:.0f}%"
        act = "-" if r.active_share != r.active_share else f"{100 * r.active_share:.0f}%"
        out(f"  {label:<34} {_fmt(r.price_day, 4):>10} {_fmt(r.price_min, 4):>10} {_fmt(r.buy20, 4):>10} {_fmt(r.buy200, 4):>10} {_fmt(r.units_day, 0):>10} {_fmt(r.gold_day, 0):>13} {_fmt(r.sold_low_day, 0):>12} {turn:>8} {act:>6} {r.hours:>5}")
    if sales:
        allh = look[look["house"] == house]
        seen = allh["sold_low_day"].notna()
        if seen.any():
            sel = allh[seen]
            out(f"  Visible sales, house {house}, {look_day}: {100 * (sel['active_share'] > 0).mean():.0f}% of the {len(sel):,} items showed at least one sale in some hour; the median item turned over "
                f"{100 * sel['turnover_day'].median():.1f}% of its stock a day and was seen selling in {100 * sel['active_share'].median():.0f}% of its hours (all lower bounds).")
            big = sel[sel["gold_day"] >= 1000]
            if len(big):
                out(f"  Among the {len(big):,} items with 1,000+ gold in play: median turnover {100 * big['turnover_day'].median():.1f}% a day, seen selling in {100 * big['active_share'].median():.0f}% of hours.")

    # ------------------------------------------------------------------ 5 crafting coverage
    recipes = read_table(store, T.KEYS["recipes"], "recipes")
    reagents = read_table(store, T.KEYS["reagents"], "reagents")
    out("")
    cov = FD.craft_coverage(look, recipes, reagents)
    if cov:
        out(f"### 5 crafting coverage on {look_day}: recipes with a product and reagents, and how many are priced on the auction house (commodity items only)")
        out(f"  {'house':>5} {'recipes':>8} {'all reagents priced':>20} {'product priced':>15} {'both':>6}")
        for h, c in cov.items():
            out(f"  {h:>5} {c['recipes']:>8,} {c['reagents_priced']:>20,} {c['product_priced']:>15,} {c['both']:>6,}")
        out("A reagent only a vendor sells has no auction price, so 'all reagents priced' is a floor. A product that is gear or a weapon trades as a regular auction, which these tables do not cover yet;")
        out("that gap is the gap between 'all reagents priced' and 'both'.")
    else:
        out("### 5 crafting coverage")
        out("Skipped: the recipe and reagent tables are not in the dataset yet (run Forever tables with save = yes).")
    return dict(text="\n".join(lines), code=code, files=files, changed=changed, houses=houses)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=4, help="UTC days of files to read (default 4; 4 also repairs up to 2 missed daily runs)")
    ap.add_argument("--end", default="", help="last UTC date to read, YYYY-MM-DD (default: today)")
    ap.add_argument("--houses", default="all", help="all (default), or house numbers separated by commas")
    ap.add_argument("--save", action="store_true", help="store the days that are new or changed (default: report only)")
    ap.add_argument("--no-sales", action="store_true", dest="no_sales", help="skip the sales classifier: prices and depth only (faster); the sales columns are left blank")
    ap.add_argument("--rebuild", action="store_true", help="new rows replace stored rows even where the stored row had more usable hours")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        print("forever_daily needs HF_TOKEN (write access to the Forever dataset to save, read access otherwise).")
        return 1
    try:
        r = run(args, FV.RetryStore(HFStore(FV.hf_repo(), tok)))
    except SystemExit:
        raise
    except Exception as e:
        print(f"forever_daily failed: {FV.scrub(e)}")
        return 1
    print(r["text"])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "forever_daily.txt"), "w", encoding="utf-8") as f:
        f.write(r["text"] + "\n")
    return r["code"]


if __name__ == "__main__":
    sys.exit(main())
