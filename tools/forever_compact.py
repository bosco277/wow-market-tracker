"""Forever compaction, COPY-ONLY: packs each completed UTC day of each house's hourly files into one ledger file. NOTHING IS EVER DELETED.

    python tools/forever_compact.py                      # report only: pack and check, store nothing
    python tools/forever_compact.py --save               # store the packs that pass the check
    python tools/forever_compact.py --save --days 3 --houses 110,116 --kinds commodity

The ledger (engine/foreverpack.py): one row per run of a listing (an auction id with unchanged values in consecutive hours), the per-hour values kept once per
hour, every column the fetcher wrote. Chosen by the player on Nov 27 after the pack probe: lossless on houses 110, 116 and 121, 3.0 to 6.2 times smaller.

For each house and kind (commodity listings, regular auctions), each UTC day that is complete (the next day started 2+ hours ago) and not packed yet, oldest first:
  1 read every hourly file of the day, every column
  2 pack it, write the bytes that would be stored, read THOSE BYTES back and rebuild every hourly file from them
  3 the pack is kept only if every rebuilt file equals the original exactly (every row, every column, every auction id)
  4 with --save, one commit per day for every pack that passed; then each stored file is read back and must be byte for byte what was sent.
Stored at packed/<source>/year=YYYY/month=MM/day=DD/ledger.parquet, outside every raw folder (no loader reads it yet). The hourly files stay where they are.
A time budget (--minutes) stops it starting new days; the rest is done by the next run. Needs HF_TOKEN (write access to save).
"""
import argparse
import hashlib
import io
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import forever as FV  # noqa: E402
from engine import foreverpack as P  # noqa: E402
from engine import forevertables as T  # noqa: E402
from engine.loader import KEY_RE, HFStore  # noqa: E402
import forever_tables as FT  # noqa: E402

MB = 1024 * 1024
KINDS = ("commodity", "auction")
SETTLE_HOURS = 2          # a UTC day is complete once the next day is this many hours old (the last hourly file lands at about 23:32)


def read_raw(data: bytes):
    """Every column of one stored hourly file."""
    import pyarrow.parquet as pq
    return pq.read_table(io.BytesIO(data)).to_pandas()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hourly_days(store, source):
    """{day: [(key, size), ...]} of the source's raw hourly files, in time order."""
    out = defaultdict(list)
    for o in store.list(f"{source}/"):
        m = KEY_RE.match(o.key)
        if m and m.group("file").startswith("hour_") and m.group("source") == source:
            out[datetime(int(m.group("y")), int(m.group("m")), int(m.group("d"))).date()].append((o.key, o.size))
    return {d: sorted(v) for d, v in out.items()}


def packed_days(store, source):
    out = set()
    for o in store.list(f"{P.PACK_PREFIX}/{source}/"):
        m = KEY_RE.match(o.key[len(P.PACK_PREFIX) + 1:].replace("/ledger.parquet", "/daily.parquet")) if o.key.endswith("/ledger.parquet") else None
        if m:
            out.add(datetime(int(m.group("y")), int(m.group("m")), int(m.group("d"))).date())
    return out


def run(args, store, now=None, reader=read_raw, writer=P.parquet_writer, preader=P.parquet_reader, commit=FT.hf_commit, clock=time.time):
    now = now or datetime.now(timezone.utc)
    t_start = clock()
    save = bool(getattr(args, "save", False))
    budget = float(getattr(args, "minutes", 90)) * 60
    max_days = int(getattr(args, "days", 60))
    kinds = [k for k in KINDS if getattr(args, "kinds", "both") in ("both", k)]
    lines = []
    out = lines.append
    code = 0
    avail = [h for h in FV.houses_in_store(store) if h not in T.EMPTY_HOUSES]
    houses = [h for h in FV.parse_houses(getattr(args, "houses", "all"), avail) if h in avail]
    last_day = (now - timedelta(hours=SETTLE_HOURS)).date() - timedelta(days=1)
    out("### forever compact (COPY-ONLY: nothing is ever deleted)")
    out(f"run at {now.strftime('%Y-%m-%d %H:%M')} UTC; houses {', '.join(map(str, houses)) or 'none'}; kinds {', '.join(kinds)}; complete days up to {last_day}; "
        f"at most {max_days} day(s) and {budget / 60:.0f} minutes; " + ("SAVE" if save else "REPORT ONLY (save = no: packs are made and checked, nothing is stored)"))
    out("A pack is kept only if every hourly file rebuilt FROM THE BYTES THAT WOULD BE STORED equals the original exactly (every row, column and auction id).")
    if not houses:
        out("Nothing to do: no house to read." if avail else "The sources file could not be read, so no house is known.")
        return dict(text="\n".join(lines), code=1, stored=[], failed=[])

    todo = defaultdict(list)                     # day -> [(house, kind, source, [(key, size)])]
    backlog = defaultdict(int)
    for h in houses:
        for kind in kinds:
            src = FV.source_name(h, kind == "commodity")
            have, done = hourly_days(store, src), packed_days(store, src)
            for d, files in have.items():
                if d <= last_day and d not in done:
                    todo[d].append((h, kind, src, files))
                    backlog[(h, kind)] += 1
    days = sorted(todo)
    out(f"Days waiting to be packed: {len(days)}" + (f" ({days[0]} to {days[-1]})" if days else "") + f"; house-kind-days: {sum(backlog.values())}.")
    out("")
    stored, failed, totals, passed = [], [], dict(raw=0, packed=0, rows=0, runs=0), 0
    days_done = 0
    for d in days[:max_days]:
        if clock() - t_start > budget:
            out(f"Time budget reached: stopping before {d}. The next run carries on from there.")
            break
        out(f"### {d}")
        out(f"  {'house':>5} {'kind':<10} {'files':>5} {'stored':>9} {'packed':>9} {'smaller':>7} {'rows':>10} {'runs':>9} {'lossless':>8} {'secs':>5}")
        day_files = {}
        for h, kind, src, files in todo[d]:
            t0 = clock()
            try:
                frames = [reader(store.read_range(k, 0, s)) for k, s in files]
                raw = sum(s for _, s in files)
                runs, meta = P.pack(frames, [k for k, _ in files])
                data = writer(runs, meta)
                runs2, meta2 = preader(data)                          # the check runs on what would actually be stored
                ok, why = P.check(frames, P.unpack(runs2, meta2))
                rows = sum(len(f) for f in frames)
                out(f"  {h:>5} {kind:<10} {len(files):>5} {raw / MB:>7.2f}MB {len(data) / MB:>7.2f}MB {raw / max(len(data), 1):>6.1f}x {rows:>10,} {len(runs):>9,} "
                    f"{'yes' if ok else 'NO':>8} {clock() - t0:>5.0f}" + ("" if ok else f"   <- NOT kept: {why}"))
                key = P.pack_key(src, d)
                assert key.startswith(P.PACK_PREFIX + "/") and not key.startswith(src)
                if ok:
                    passed += 1
                    day_files[key] = data
                    totals.update(raw=totals["raw"] + raw, packed=totals["packed"] + len(data), rows=totals["rows"] + rows, runs=totals["runs"] + len(runs))
                else:
                    failed.append((h, kind, str(d)))
                    code = 1
                del frames, runs, runs2
            except Exception as e:
                failed.append((h, kind, str(d)))
                code = 1
                out(f"  {h:>5} {kind:<10} FAILED: {FV.scrub(e)[:300]}")
        days_done += 1
        if save and day_files:
            try:
                commit(day_files, f"forever compact: {d} ({len(day_files)} pack(s), copy-only)")
                bad = [k for k, v in day_files.items() if sha(FT.read_key(store, k) or b"") != sha(v)]
                if bad:
                    code = 1
                    out(f"  stored, but {len(bad)} file(s) did NOT read back identical: {', '.join(bad)}")
                else:
                    out(f"  stored {len(day_files)} pack(s) in one commit; each read back byte for byte identical.")
                stored.extend(k for k in day_files if k not in bad)
            except Exception as e:
                code = 1
                out(f"  UPLOAD FAILED for {d}: {FV.scrub(e)[:300]}")
        elif day_files:
            out(f"  {len(day_files)} pack(s) passed; report only, nothing stored.")
    left = len(days) - days_done
    out("")
    out("### summary")
    if totals["raw"]:
        out(f"  packed {totals['raw'] / MB:,.1f} MB of hourly files into {totals['packed'] / MB:,.1f} MB ({totals['raw'] / max(totals['packed'], 1):.1f}x smaller); "
            f"{totals['rows']:,} listing-hours kept as {totals['runs']:,} runs.")
    out(f"  passed the check: {passed} pack(s); stored: {len(stored)}" + ("" if save else " (report only)"))
    out(f"  failed the check or could not be read: {len(failed)}" + (f" ({', '.join(f'{h} {k} {d}' for h, k, d in failed[:20])})" if failed else ""))
    out(f"  days still waiting after this run: {left}")
    out("  Hourly files deleted: 0 (this job never deletes; deleting is a separate decision, not built).")
    return dict(text="\n".join(lines), code=code, stored=stored, failed=failed, passed=passed, days_done=days_done, days_left=left)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--houses", default="all", help="all (default) or house numbers separated by commas")
    ap.add_argument("--kinds", default="both", choices=["both", "commodity", "auction"])
    ap.add_argument("--days", type=int, default=60, help="at most this many days in one run, oldest first (default 60)")
    ap.add_argument("--minutes", type=float, default=90, help="stop starting new days after this many minutes (default 90)")
    ap.add_argument("--save", action="store_true", help="store the packs that pass (default: report only)")
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        print("forever_compact needs HF_TOKEN.")
        return 1
    try:
        r = run(args, FV.RetryStore(HFStore(FV.hf_repo(), tok)))
    except Exception as e:
        print(f"forever_compact failed: {FV.scrub(e)}")
        return 1
    print(r["text"])
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "forever_compact.txt"), "w", encoding="utf-8") as f:
        f.write(r["text"] + "\n")
    return r["code"]


if __name__ == "__main__":
    sys.exit(main())
