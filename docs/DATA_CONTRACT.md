# WoW Market Engine: Data Contract (v0.1.0, classifier v1.0)

This is the reference for what goes into the engine, what comes out, and what each number does and does not mean.
Retail commodities are the test bed. Forever plugs in through the same `Snapshot` contract.

**Read section 7 first if you only read one thing.** Every accuracy number in this repo comes from a simulator whose
market behaviour I invented. None of it is validated against real sales yet.

---

## 1. Layers

| Layer | What it holds | Where it lives | Rebuildable? |
|---|---|---|---|
| **Bronze** | Raw auction snapshots exactly as fetched (fetch.py v2, with `snapshot_ts` from `Last-Modified`) | B2 + HF mirror (Parquet) | No. This is the only irreplaceable layer. |
| **Events / intervals** | Per consecutive snapshot pair, per item: sold bounds, flags, event labels | derived | Yes, from bronze |
| **Item-hour metrics** | Rates, depth VWAP, sell-side clearing price, expected sales | derived | Yes |
| **Signals** | Strategy output, alerts, ROI/sizing | derived | Yes |

Rule: never store a derived number without the classifier version and parameters that produced it.
`CLASSIFIER_VERSION = "v1.0"`.

## 2. Canonical snapshot contract (`engine/schema.py`)

A `Snapshot` is one complete observation of one source at one moment.

| Field | Type | Meaning |
|---|---|---|
| `source` | str | e.g. `commodities` (retail, region-wide). One source = one market. Never mix. |
| `ts` | naive UTC `Timestamp` | The time Blizzard published the data (`Last-Modified`), **not** when we fetched it. |
| `book` | DataFrame | Canonical columns below |
| `kind` | `"commodity"` or `"auction"` | `auction` (per-listing items) is **not implemented in the classifier** and raises `NotImplementedError`. |
| `covered_items` | set or None | Items this snapshot actually covers. `None` means the whole market. Needed for partial addon scans. |
| `meta` | dict | e.g. `ts_source` when we had to fall back to `scan_timestamp` |

Book columns (`CANONICAL_COLUMNS`):

| Column | Type | Notes |
|---|---|---|
| `auction_id` | int | Unique per listing within a snapshot. Used as the tie-break for queue order. |
| `item_id` | int | |
| `variant_key` | str | Reserved for non-commodity variants (context, bonus lists). Empty for commodities. |
| `quantity` | int > 0 | Units in the listing |
| `unit_price` | int > 0, **copper** | 1 gold = 10,000 copper (`COPPER_PER_GOLD`) |
| `time_left` | `SHORT` / `MEDIUM` / `LONG` / `VERY_LONG` | |

`coerce_book` rejects duplicate ids, non-positive quantity or price, and unknown `time_left` values with `SchemaError`.
Failing loudly is deliberate: a silent bad row corrupts every later number.

### fetch.py v2 → Snapshot (`snapshots_from_blizzard_v2`)

- Groups by `(source, snapshot_ts)`. Falls back to `scan_timestamp` if `snapshot_ts` is missing and records `meta["ts_source"]`.
- Commodities: `id`→`auction_id`, `item.id`→`item_id`, `unit_price` as is.
- Realm auctions: `unit_price = buyout / quantity`; rows with only a bid (no buyout) are dropped.
- Covered by `tests/test_pipeline.py`: simulator → v2-shaped frame → adapter → classifier gives identical output to the direct path.

### `time_left` buckets: UNVERIFIED

`TimeLeftBuckets` defaults: SHORT 0–30 min, MEDIUM 30 min–2 h, LONG 2–12 h, VERY_LONG 12–48 h.
These are from memory of the Blizzard docs and have not been checked. They only affect the expiry-feasibility part of
the **low** bound. Verify them before trusting low bounds. Everything is parameterised so correcting them is one edit.


## 2b. Loader (`engine/loader.py`) and item picker (`engine/picker.py`)

- **Where it reads from:** Hugging Face by default. `fetch.py` mirrors every hourly file there and compaction only
  deletes from B2, so HF keeps the hourly files and has no daily download cap. B2 is supported but metered: a byte
  budget (default 0.8 GB per UTC day, remembered between runs) stops the load *before* it would go over.
- **Never downloads twice:** each file is cached (already trimmed) in `.lake_cache`. The cache is disposable.
- **Plan first:** `plan_load` / `tools/lake.py plan` reports files, MB, cached files and missing days without downloading.
- **Layouts handled:** v2 hourly, v1 hourly (no `snapshot_ts`; falls back to fetch time and is flagged
  `meta["ts_source"]="scan_timestamp"`), and compacted `daily.parquet` (`clean_item_id`, `item_json`). Hourly files are
  preferred when both exist. Duplicate snapshots are dropped. A file with the wrong columns is an error, never a guess.
- **Item filter:** `item_ids=[...]` loads only those items and marks snapshots `covered_items`; classifier output for
  those items is identical to a full load (tested).
- **Not yet tested against real files:** reading actual Parquet (needs `pyarrow`), the real Hugging Face and B2
  responses. Tests use stand-in files. First live run is the real test: use `plan_only`, then the full run.
- **Picker:** chooses a spread of slow/medium/fast items with steady prices, deep books and small cost, and sizes the
  purchase at about two "normal hourly sales" so it stands out in the data. See `docs/EXPERIMENT_PROTOCOL.md`.


## 2c. Item report and flip finder (`engine/itemview.py`)

- `tools/lake.py item --ids ...`: hourly lowest price, units at the floor / within 10% of it / listed, new units, and the
  sold_low / sold_est / sold_high bounds for one or more items.
- `tools/lake.py flips`: three layers, each stricter than the last (v0.4.2; the first real run showed layer 2 alone is far too loose):
  1. **Jump:** the item's lowest price rose by 30% or more AND at least 80% of the units below the new lowest price are gone.
  2. **Buyout:** those units could not have expired (LONG / VERY_LONG time left). First real run: 4,538 of these in 3 days
     across 10,066 items, about 75 an hour, 44% on items under 1 gold. That is ordinary buyers eating a thin cheap level,
     not flippers, so layer 2 is NOT a flip count.
  3. **Likely flip:** also at least 3x the item's normal hour (75th percentile of its estimated hourly sales) AND at least
     half of the removed units reappear as NEW listings within 25% of the new lowest price in the same interval.
     Heuristic and not yet validated: the one known real flip (Mystic Sapphire, Thu 10-29 22:33 ET, 1.49 -> 197.00 gold) is
     the first thing to check against it.
  - **Follow-up only counts observed time.** `ended_by` is `reverted` (price came back within 10% of the old level),
    `hole` (our data stopped for over 2 hours first: return time UNKNOWN, left blank), or `end_of_data` (still up at the
    last snapshot). v0.4.1 ignored holes: 1,319 of 4,538 events had follow-up across the 21-hour hole, which inflated the
    revert times and gave nonsense volumes (median "units sold while elevated" 493 for those vs 37 for the rest).
  - Revert statistics use `revert_share_within(events, hours)`: only events watched at least that long (or reverted
    sooner) are counted, so a hole or the end of the data is never counted as "still up" or as "came back".
- **This describes past flips. It is not a recommendation to flip.** Before any flip recommendation is trustworthy we need:
  (1) how often and how fast elevated prices get undercut or sit unsold, measured from many past flips; (2) the true cost of
  the buyout sweep (exact partial-tier VWAP, already built); (3) the effect of newest-first ordering: a rival who lists at
  your exact price sells BEFORE your units, so an elevated price is easier to attack than under oldest-first; (4) real
  sell-through from your own experiment. Outputs would be a range (low / est / high), never one profit number.

### Holes in our own data

**CONFIRMED CAUSE of the "21-hour hole" (v0.4.5).** `inspect` on the real files showed the Oct 29 daily file holds all its
hours (continuous hourly fetch times from Wed 10-28 21:01 ET to Thu 10-29 16:01 ET, then publish times from 16:33). It mixes
layouts: 5,255,352 of its 6,046,896 rows (the fetch-time hours) have an empty `item_id`, `snapshot_ts`, `source` and
`variant_key` but a full `clean_item_id`. Two loader bugs removed them: (1) `clean_item_id` was ignored because an
`item_id` column existed; (2) even with the id recovered, rows with an EMPTY `source` were silently dropped by the grouping
step. Real gaps are only the short ones (Tue 10-27 09:31-12:01 ET, Wed 10-28 17:01-19:50 ET). Also: the first 23 snapshots
of the Oct 26 file (Sun 10-25 20:31 to Mon 10-26 18:31 ET) hold 13-31k listings where a full book has ~265k (partial
fetches); the loader now treats any snapshot under 50% of the typical listing count as missing and says so.

**Mixed-layout days (v0.4.4):** a compacted daily file can mix older hours (item id in `clean_item_id`, no `snapshot_ts`)
with newer hours (`item_id`, `snapshot_ts`). v0.4.3 and earlier read `clean_item_id` only when the file had NO `item_id`
column, so on a mixed day every older row was dropped without a warning. The loader now takes the item id row by row from
`item_id`, then `clean_item_id`, then the `item` struct, and prints a warning if any rows still have none. Cached files
from older versions are ignored (cache names now end in `.v2.pkl`). `tools/lake.py inspect` (workflow mode "inspect")
downloads each daily file and prints its columns, empty cells and one line per snapshot time. Status: the bug is proven in
a test; that it caused the Oct 28-29 hole in the real data is NOT yet confirmed.

**Loader rule (v0.4.3):** per UTC day the loader uses the hourly files, and ALSO that day's daily file when the hourly files
cover fewer than 22 of the 24 hours. v0.4.2 and earlier used the daily file only when the day had NO hourly files, so
hourly files that were partly deleted hid hours the daily file still held. Snapshots stored twice are merged when they are
within 10 minutes of each other. `tools/lake.py files` (workflow mode "files") lists what exists per day, downloading nothing.

`pick` and `flips` print the holes (over 2 hours with no snapshot). First real data had one of about 21 hours (Wed 10-28
~17:00 ET to Thu 10-29 ~13:00 ET) plus two shorter ones. The classifier leaves hole intervals out of every rate
(`flag_long_gap`), so the picker looks back 72 h (was 48) and needs 50% coverage (was 60%); with the hole the 48 h window
had 0.52 coverage, which rejected every item as "too little data".

## 2d. Market rate, price walls and profiles (`engine/market.py`, v0.5.0)

Built on LISTINGS only, so it should carry over to any market whose auction house publishes its book. Sales rates come
from the classifier as low / best / high units per hour, are a floor, and are poor for very fast items (see section 7).

- **Price bands** (`lake.py market`): percentiles of the hourly lowest price with short spikes and dips left out
  (`flag_spikes`: 1.5x the centred 25-hour median, episodes up to 10 hours; longer moves are a new level, not a spike).
  Reports where the current price sits in its own range. The first real example of why: Sorrowmoss on TSM shows one ~99 gold
  spike that makes a single Friday 4-5 PM slot look like a weekly peak.
- **TSM-style market value** (baseline, from our own book): cheapest 30% of auctions, cut at the first 20%+ jump after the
  first 15%, drop prices 1.5 SD from the mean, average. TSM then smooths over 14 days (half-life ~2 days); we do not. TSM's
  public CSV (`public-data.tradeskillmaster.com/retail/us/commodities.csv`) has marketValue, minBuyout, recent, historical
  and NO sales fields, so it can validate prices, not sales.
- **Price wall:** walk up the book until `sales rate x horizon` units are added; that price is the wall. Reported at the
  low / best / high rate. A listing at the wall waits `units at LOWER prices / (sales rate - new cheaper listings per hour)`
  under newest-first selling (same-price units are behind you). New cheaper listings keep joining the queue ahead of you,
  so the no-inflow wait is a best case. Measured net drain is not affected by the blind spot (a listing posted and sold
  inside one hour adds and removes the same units), which is why `reach_stats` (history of how often and how fast the lowest
  price reached a level, refills included) is the better tool for very deep markets.
- **Two flags per item:** `deep_market` (units at or below the wall >= 20x your trade size) and `velocity_matters` (False
  when `trade_units` listed at the wall would FULLY sell within 12 h even at our LOW rate, so a precise rate would not
  change the decision). v0.5.1: the time to sell counts your own units too (`sell_hours_*`); before, only the units ahead
  of you were counted (`wait_*`, the time to reach the front of the line), which made most items look velocity-insensitive.
  Real examples: Linen Cloth (about 18,800 units below a 9 silver wall) and Eternal Crystal (about 1,960 below 2.29 g)
  are both velocity-insensitive although our rates for them are 14-26x below TSM's.
- **Hour-of-week profile** (`lake.py profile`): medians of price, listed quantity and estimated sales by hour of day, plus a
  day-of-week x 4-hour grid. Every profile states how many weeks it rests on and is a DRAFT until there are 3 or more;
  with one week each day-of-week slot has one observation. Quantity shows a stable weekly shape faster than price does.
- **Flip shelf** (`find_shelf`): the units below a 30%+ price gap, total cost under a cap, net margin (relist price x 0.95 /
  average paid - 1) of 25%+. Relist at the next tier (newest-first puts you first there). A snapshot is up to an hour old
  and fast items lose shelves in minutes (the real Eternal Crystal shelf of 90 units at 0.99-1.00 was gone within the hour),
  so this is a candidate list to check live, not a signal to act on.

## 3. What the API can and cannot tell us

The Auction House API publishes **listings, not sales**. There is no receipt. Everything below is inference from how the
book changes between two snapshots.

Facts the classifier relies on (commodities):

1. A listing whose quantity **dropped** was partially bought. That is a certain sale (`sold_cert`).
2. Buyers take cheapest-first, so bought units form a **prefix of the previous book** ordered by `(unit_price, auction_id)`.
3. A listing that vanished **beyond the first untouched survivor** in that order cannot have been bought. It expired or was cancelled.
4. A listing that vanished **inside** the sweep was probably bought, but could have been cancelled, or expired if its `time_left` allowed.
5. Within one price tier the real fill order is unknown. The default (`fifo_within_tier=False`) treats the tier as unordered, which protects the high bound if the market is not FIFO. Costs almost nothing if it is.

### The blind spot

A listing **created and sold between two snapshots** never appears in either. Observed sales are therefore a **floor**
on true volume. In the simulator this is the dominant error on hot items (see section 7).


### Verified: newest listing at a price is bought first (LIFO), retail commodities

- **Test:** `tools/lake.py order` (`engine/ordercheck.py`). For each price tier where one listing was *partly* bought (a
  certain sale), did the newer or the older listings at that price disappear? Listings that could have expired are ignored.
- **Result on the real retail data (Oct 27-29, 2026):** 40,608 usable cases: 31,228 LIFO pattern (77%), 374 FIFO pattern
  (0.9%), 9,006 mixed (22%; cancels and relists muddy some tiers). Verdict: LIFO.
- **Outside corroboration (player reports, not official Blizzard statements):** The Lazy Goldmaker (Sep 30, 2026, from the
  Forever beta) says Forever uses the retail AH and that at the same price "he will get the 50 that were posted most
  recently"; a 2024 forum post says retail needs no undercutting because of "last one in = first one out".
- **Forever itself is NOT verified from data.** Re-run the same check on the first days of Forever data before relying on it.
- **What it changes:** `sell_clearing_price(..., tier_order="lifo")` and `BacktestConfig(tier_order="lifo")`: joining a
  price puts you ahead of everything at that price, so only strictly cheaper listings are ahead of you and no undercut is
  needed. **What it does not change:** the classifier stays on `tier_order="unknown"` (see `tools/lifo_check.py`: knowing
  the rule changes coverage by about 0.002 in a newest-first market, while a wrong guess pushes 2-4% of intervals above the high bound).

## 4. Classifier output (`engine/classifier.py`)

`classify_transitions(prev, curr, params)` → `TransitionResult(source, t0, t1, gap_minutes, events, items)`.

Per item row (`ITEM_COLUMNS`), the ordering invariant is always:

```
sold_cert <= sold_low <= sold_est <= sold_high
```

| Column | Meaning |
|---|---|
| `sold_cert` | Units from partial fills. Certain. |
| `sold_low` | `cert + max(0, infeasible_units − relist_attribution × relist_units)`. "Vanished inside the sweep and cannot have expired." Minus an allowance for undercut-cancel-relist. Conservative. |
| `sold_est` | `cert + Σ sweep × (1−λ) × (1−cancel_share)`, where λ is the uniform-remaining-time expiry fraction. Clipped into [low, high]. A best guess, not a bound. |
| `sold_high` | `cert + sweep + tie`. Everything that plausibly could have been bought. |
| flags | `flag_wipe` (>= `wipe_threshold` of units gone, >= `wipe_min_units`), `flag_long_gap` (gap > `max_gap_minutes`), `prefix_violations`, `anomaly_increases` |

Event labels: `PARTIAL_FILL`, `VANISHED_IN_SWEEP`, `VANISHED_TIE`, `VANISHED_BEYOND_SWEEP`, `QUANTITY_INCREASED`.

`aggregate_rates(intervals, now, window_hours)` → per item daily low/est/high scaled by **observed** hours, plus `coverage`
(observed fraction of the window) and `n_intervals`. Flagged intervals are the caller's choice to exclude.

### Parameters

| Parameter | Default | Calibrated? |
|---|---|---|
| `buckets` | see section 2 | **No, unverified** |
| `fifo_within_tier` | False | Chosen from simulator tests |
| `cancel_share` | 0.10 | **No. Invented.** Estimate from your own buy/list experiment. |
| `relist_attribution` | 1.0 | **No. Invented.** |
| `wipe_threshold` / `wipe_min_units` | 0.9 / 50 | Heuristic |
| `max_gap_minutes` | 240 | Heuristic |

## 5. Metrics (`engine/metrics.py`)

Buying and selling need different assumptions, on purpose:

- **Buy side: be conservative about liquidity.** Use the HIGH sales end, plus `blind_spot_correction`, with a `safety_buffer`
  via `target_depth(expected_units, blind_spot_correction, safety_buffer)`. Overestimating what will sell makes you size too large.
- **Sell side: be conservative about your fill.** Use LOW/EST sales and your queue position via
  `sell_clearing_price(item_book, expected_sales_units, my_units)`. It joins the deepest price tier that still leaves room
  for the expected sales ahead of you, undercuts when the floor tier is too deep, and reports a partial fill when the
  market will not absorb your whole stack. It never quotes above the highest ask.
- **Depth VWAP** is exact, with partial tiers (`depth_vwap`), not snapped to price walls. `snap_to_tier=True` exists only for comparison; it makes VWAP jump at tier boundaries.
- `AH_CUT = 0.05`. `round_trip_roi(buy, sell)` = `(list_price × 0.95 − buy_vwap) / buy_vwap`.
- `expected_sales(...)`: trailing-window rate, optional same-hour-of-week seasonal component, blend by `seasonal_weight`.
- `daily_volume_quantile(...)`: distribution of daily volumes over a lookback.

**Not a TSM clone.** There is no "sold per day" number from a public source we could validate against; TSM's method is unpublished.

## 6. Backtest rules (`engine/backtest.py`)

- **No look-ahead.** At snapshot k the strategy sees snapshots[0..k] and intervals ending at or before k. `tests/test_backtest.py` perturbs future data and checks earlier decisions do not change.
- **Entry** sweeps the observed ask book at the decision snapshot (you pay real tiers). Optimistic about latency: someone else may buy first.
- **Exit** lists at the queue-position price given expected absorption over `exit_horizon_hours`, net of the 5% cut. Units not expected to sell are valued at `unfilled_haircut` × floor × 0.95.
- **One open position per item.** Cash is tracked; entries fit the cash on hand. Profits are reinvested.
- **P&L is a range.** `run_backtest_range` runs low/est/high absorption and reports all three. Report the range, never one number.
- `DipBuyer` is a reference baseline to beat, not a recommendation.

A simulated P&L says the **machinery** works. It says nothing about whether real markets pay.

## 7. Validation status and honest limits

Method: the simulator (`engine/simulator.py`) runs a market with a ledger of true sales, cancels and expiries. We classify its
snapshots and compare to the ledger (`engine/validation.py`, `tools/validate_classifier.py`).
3 seeds x 96 h, default classifier:

| Item (profile) | coverage | high_viol | est/true | low/true | true/observable (blind spot) |
|---|---|---|---|---|---|
| 1001 staple | 0.965-0.993 | 0 | 0.947 | 0.699 | 2.9x |
| 1002 mid | 0.965-0.993 | 0 | 0.981 | 0.774 | 2.4x |
| 1003 thin | 0.965-0.993 | 0 | 1.107 | 0.986 | 1.14x |
| 1004 very liquid | 0.965-0.993 | 0 | 0.943 | 0.690 | 3.75x |

(`est/true` and `low/true` are measured against sales that were observable in principle, i.e. excluding the blind spot.)

What this does show:
- The bound logic is internally consistent. `high` was never below true observable sales in these runs, and is robust to a non-FIFO market once tier-aware.
- **The blind spot dominates.** On liquid items true volume is roughly 2.4-3.75x what snapshots can see. Classifier cleverness matters less than snapshot frequency and the blind-spot correction.
- Under heavy cancel/relist churn (6x cancel rate, 90% relist), the conservative bound degrades: low-bound violations rose from about 2-3.5% to 3-7%, and the estimate overshot up to 2.2x on the thin item.
- Missed snapshots and shifted `time_left` buckets barely moved results. A demand spike was handled.

What this does **not** show:
- That real players behave like the simulator. Cancel rates, undercut habits, whale behaviour and bot activity are my guesses.
- Anything about Forever.
- That any strategy is profitable.

### Untested in this sandbox
Real Parquet read/write, DuckDB, boto3 against B2, Hugging Face commits. The sandbox blocks PyPI, so those were stubbed.

## 8. Forever adapter checklist

When Forever data exists:
1. Confirm there is an AH API at all, and whether it is commodity-style (per-unit price tiers) as the beta suggested. Unconfirmed.
2. Confirm what `Last-Modified` / snapshot time means. If data comes from addon scans, populate `covered_items`; a partial scan must never be read as "listings vanished".
3. Re-verify `time_left` buckets for the new game.
4. Realmless: decide what `source` is. One market per source.
5. If listings are per-auction rather than per-unit, implement `kind="auction"` in the classifier first.
6. Re-run the buy/list experiment to recalibrate `cancel_share` and the blind-spot correction. Do not carry retail values over.
7. Non-commodity (gear, weapons, transmog) is a deferred analysis, but the RECORD must be complete from the first Forever hour. Today `fetch.py` stores, per
   realm auction per hour: Blizzard's auction `id` (the key for following one listing across hours), the whole `item` struct untouched (id, context, bonus_lists,
   modifiers, pet fields), `bid`, `buyout`, `quantity`, `time_left`, `snapshot_ts` and `scan_timestamp`; compaction keeps all of it (`item` becomes `item_json`).
   On launch day confirm for Forever: the realm/connected-realm ids and namespace to fetch, that each of those fields exists, and the size of one hourly realm
   file (check it against the B2 cap and the compaction memory before adding realms). Missing fields cannot be recovered later.
8. Forever has NO realms (per AHledger's explainer, citing Blizzard): 18 markets = 3 rulesets (Normal, PvP, RP) x 2 factions x 3 regions (US, EU, CN), plus neutral
   auctioneers (Gadgetzan, Booty Bay, Everlook; 15% cut, faction houses 5%); listings run 12 / 24 / 48 hours. `source` should be the market id (for example
   `forever.normal.alliance.us`). Unverified: whether Forever has commodity-style stacks at all.
9. NO-API FALLBACK. UPDATE Nov 1, 2026: our own `api_probe` found `dynamic-forever-us` (see 2m): Blizzard knows the namespace and lists 51 realms; static data, the connected-realm index, commodities and the auction-house index were not reachable. As of Oct 26-28 independent probes found no Forever namespace on the Battle.net API (403 for every guess), and the Classic Era / Anniversary auction
   endpoints return 404 (index only, no listings), so Forever may never expose listings. Backup sources, in order: (a) the free AHledger API (api.ahledger.com, no key,
   300 requests/min, aggregates per market: lowest price, median, quantity, auction count; per-scan points for 24 h, hourly buckets for 7 d, daily for 30 / 90 d; terms:
   credit AHledger with a link wherever its data is shown, do not present it as our own scan or resell it as a feed); (b) our own scans through an addon + SavedVariables
   reader (the game allows one full scan per 15 minutes, only with the auction window open; the file is written at logout / reload, so a few updates a day per player);
   (c) pooling scans from friends or guildmates. Sales cannot be inferred from totals alone (see the Undermine check), so listing-level scans matter most for calibration.
10. LAUNCH-DAY PROBES (Market tools, no files saved): `api_probe` asks Battle.net whether a Forever namespace exists (guesses include camelot, forever, classicplus; add
   more as words in item_ids) and, if one answers, which of the realm / auction / commodity paths work and what fields they return; `ahledger_check` lists AHledger's
   endpoints and the shape of their answers. Run `api_probe` before launch and again on Nov 4 and on the days after; run `ahledger_check` once now. Details in 2m.
11. FOREVER LAUNCH PLAYBOOK (what the engine does with the first days, details in 2o). (a) Get ANY data source running from hour 0 (API if `api_probe` finds a namespace;
   otherwise AHledger and/or our own addon scans) and run `daily` every day from launch, so item ages count from launch. (b) From day 1 run `launch_now`: a settled-price and
   settled-stock RANGE per young item from the retail launch prior (`launch_backtest`, stored in engine/launch_prior). (c) Around day 60, when Forever items have their own
   settled level (median over days 28-56), score the retail prior against Forever's own items exactly as `launch_backtest` scores held-out launches; the hit rates say how far
   the ranges can be trusted and whether to widen them. (d) Until then the ranges are a retail prior on an economy that starts empty: treat the SHAPE (steep first week, flat
   by week 5-6) as likely and the LEVELS as unproven.

## 9. Next steps (in order)

1. Loader: Parquet lake (B2/HF via DuckDB) → `Snapshot` list.
2. Choose experiment items from your own v2 data (liquid enough to sell, cheap enough to risk).
3. Buy/list experiment protocol and log template. Real sales are the ground truth we lack.
4. Fit `cancel_share` and the blind-spot correction from that log; re-run validation on real data.
5. Verify `time_left` buckets.
6. Crafting margins, nightly derived job, Discord alerts.

## 2e. v0.5.2: profile guards, feed blips, reconcile

- **Feed blips** (`market.flag_blips`): a snapshot where an item's units listed fall to 30% of normal or less for at most two
  snapshots and are back above 70% right after. Real buyouts last hours. Known example: Eternal Crystal Thu 10-29 18:33 ET
  (about 685k units -> 63,361 -> 685k); Undermine's chart shows the same drop, so it comes from the feed, not from our loader.
  An episode still open at the end of the data is not flagged. Blip hours are left out of the profile (price, quantity, sales).
- **Profile columns:** `price_vs_level` (each hour against that day's own centred 25-hour median, so a week-long trend is not
  read as an hourly effect) and `sold_share` (the slot's share of the day's estimated sales, sums to 1 within a day; hourly
  sales are lumpy and the mean of a few days is dominated by one big hour). `price_median` stays as the raw price.
- **Spike rule:** a move smaller than 5 silver (`min_move_gold`) is never a spike; on a 4-8 silver item one price step is
  25-100%, and Linen Cloth lost 34 of 91 hours to this before.
- **Reconcile** (`engine/reconcile.py`, `tools/lake.py reconcile --log experiment_log.csv`), v0.5.5. Per log row: the snapshot
  before and the first snapshot after posting. Findings from the first real run (47 listings) shaped it:
  - Detection and the blind spot are only measured for listings with NOTHING else at their price (`alone`). Otherwise other
    sellers' units arriving and leaving at the same price swamp the signal (first run: negative "seen" shares, and a blind
    spot of 25% that is really 1 of 6). For `alone` listings the next snapshot showed 100-113% of the units posted.
  - Timing (`still_h` / `gone_h`, a window because snapshots are hourly) is only measured for listings that were also the
    CHEAPEST in the book (`cheapest_at_posting`); it is compared with the hours the classifier's recent rate predicts.
  - `at_front_%` is the share of the next snapshots in which nothing was cheaper than the listing. An unsold listing that
    stayed at the front all 12 hours and was predicted to sell is evidence our rate is overstated; one that was pushed back
    by cheaper listings says nothing about the rate.
  - Queue order (newest-first or not) is NOT read from the log: SAME listings share a price with older units, which
    snapshots cannot tell apart. The `order` check on the book history does that.
  Rows posted with a stale "before" snapshot are marked unusable beyond 90 minutes. Rows whose notes say "abnormal" are kept
  out of the rate check. The log reader accepts tab or comma separated files, "Note 1", item names when item_id is #N/A,
  below/above lane words, and flags batches whose prices disagree by more than 10% (a likely typo).

## 2f. Flip shelves (engine/shelves.py, `shelves` table from `derive`)

One row per item at most: the best shelf of cheap units below a price gap (gap >= 15%), bought cheapest-first up to the unit
and gold limits (`--max-units`, `--max-position-gold`; defaults 300 units, 150 gold), relisted at the tier above the gap
(newest-first puts you at the front of that tier). Columns: units, shelf_units, cost_gold, avg_price_gold, target_gold,
breakeven_gold (= avg / 0.95), net_margin (at target), net_margin_stress (selling 10% under), profit_gold, profit_stress_gold,
sell_hours_low/est/high (units ahead of you after a partial buy + your own units, divided by the sales rate seen while the
price was at the target; falls back to the overall rate, `rate_from_level` = False), cheap_units_ahead, book_share, six
tests (test_price, test_speed, test_size, test_margin, test_regime, test_book = pass / warn / fail) and `verdict` (candidate: no fail and at
most one warn; weak: no fail; no). The tests describe prices and rates; they cannot see other players' intentions.

test_regime compares the cost and the target with the item's longer history (median of all hourly floors we hold, up to 8 days;
long_median_gold, long_p10_gold, long_p90_gold, long_days): fail if avg cost > 1.25x the median, or the last 48 h median is
>= 1.3x the median and the price is now below 0.85x the last 48 h p90 (a market reset in progress), or the target > 1.5x the
median; warn if the target > 1.25x the median or under 3 days of history. cost_vs_long and target_vs_long are the ratios.
The `shelves` table also carries daily_low / daily_est / daily_high (units sold per day) of the item.

## 2g. Real results from Journalator (engine/journalator.py, `tools/lake.py ledger`)

Reads the add-on's saved file `WTF/Account/<ACCOUNT>/SavedVariables/Journalator.lua` (type /reload in game first; the add-on archives
its logs then). The logs sit in `JOURNALATOR_ARCHIVE` as LibSerialize -> raw DEFLATE -> LibDeflate "EncodeForPrint" text; the decoder
is checked against a real file (counts equal the CSV exports). Times are Unix seconds (UTC). A sale's time is when the mail was
collected: an upper bound on when it sold. Sale invoices carry no item link, only the name, so sales are matched to listings by
name and price.
- invoice (buyer): value = total paid. invoice (seller): value = gross, consignment = the 5% cut, deposit = deposit returned;
  money received = value - consignment + deposit (verified on 20 sales).
- an expired listing forfeits its deposit; a sold unit returns it.
- Tables: purchases, sales, postings, failures; `match_listings` (units sold / expired / open per posting, hours to collect, deposit
  lost); `flip_ledger` (per item bought: profit on sold units after the cut, unsold stock at cost, deposits).

## 2h. Item groups (engine/groups.py, `tools/lake.py groups`, table `engine/item_groups.parquet`)

Built from Blizzard's item endpoint (`/data/wow/item/{id}`, namespace static-us), the same call the dictionary job makes. One row
per item: item_class / item_subclass (plain words and ids), quality, item_level, vendor_sell_gold, vendor_buy_gold, and our tags:
`group` ("Crafting materials: Cloth"), `family` (coarse parent), `use` (finer label guessed from the NAME where Blizzard lumps items,
for example Healing potion, Feast / cauldron, Ore or bar; a guess), `siblings` / `tier_guess` (items sharing a name are quality tiers;
tier_guess ranks them by item id and is a guess). Only ids not yet in the table are fetched, so later runs are short. The summary the
command prints gives, per group, items, median price, median sales per day and `spread` (p75 / p25 of sales per day): a group is only
useful for extrapolation if its items behave alike, so spread near 1 is good and a large spread means the group is too coarse.
Community groups (TSM public groups) are not used: they are partial and personal.

### 2i. Group test (engine/grouptest.py, `tools/lake.py grouptest`; reads engine/items + engine/item_groups, writes nothing back)

Leave-one-out: every item is guessed from the OTHER items in its group (median in log space, at least 3 peers), and the same item is
guessed from its family and from all items. The miss is reported as a multiple (x2.0 = about double or half) for sales per day and for
price, scored on every item and again on Medium/High-confidence items only. Verdict per group: "too few to judge" (< 6 scored),
"no better than guessing" (miss not at least 15% smaller than guessing from all items), "helps, family is as good", or "adds to its
family". Also printed: how higher quality tiers compare with the lowest tier of the same name; whether a same-name sibling guesses
better than the group; and, for big groups with spread >= 10, which extra split (item-id band as a stand-in for expansion, quality,
has-tiers, name use) lowers the miss most. Splits tried on the coarse groups: `era` (item-id range, old < 120k, mid 120-200k, previous 200-240k, current 240k+; a rough guess at
the expansion, not Blizzard data), `id_band` (blocks of 20,000), `quality`, `era_quality`, `tiered`, `use`; each slice is printed with its
median sales per day and median price. A rhythm test follows: each item's pattern (share of sales by hour of day and price by hour from
engine/item_hourly, US Eastern, spike/blip/wipe/gap hours left out, items with 96+ usable hours; share of sales by day of week from
engine/item_daily, items with 14+ complete days and every weekday seen twice) is
guessed from the average of the other items in its group, family and all items and from a flat day; verdicts are "group-specific rhythm",
"same rhythm as every item" or "no better than a flat day".
Part 3, lifting together (reads engine/item_daily.parquet, see 2j): quantities drop out because each item's day is compared with ITS OWN
typical day (sales_lift = log10((units that day + 1) / (median day + 1)); price_lift likewise; days need 18+ usable hours, items 14+ days
and a median day of 3+ units). For each item the lift series is correlated with the average of the other items in its group (corr_group)
and with the average of every item OUTSIDE its group (corr_market, so a strong group cannot inflate its own control); extra = the
difference. Under 14 days of history the results are labelled PROVISIONAL (items need only 6 complete days; the median over many items averages
the noise down, so only big groups and families can be read). The day-of-week pattern waits for 14 days. Spike days (sales >= 2x the item's normal): how many times more likely its peers are to lift the same day than on a random
day, in the group (spike_group) and outside it (spike_market). Verdicts: "moves together beyond the market" (extra >= 0.10 and
corr_group >= 0.25), "moves with the whole market", "no shared movement"; "spikes together beyond the market" needs 10+ spike days.
Parts 2 and 3 are the ones that ask whether items MOVE alike; part 1 only asks whether they sell similar quantities.
Nothing here feeds a forecast; it decides which groups are safe to lean on. Planned next:
a model group per item (separate from the official group, which never changes), set only on strong, repeated evidence, every move logged.

## 2j. Daily history per item (engine/daily.py, `tools/lake.py daily`, table `engine/item_daily.parquet`)

`derive` keeps only 8 days of hourly rows (`history_hours`), too short to see whether items move together from day to day. `daily` turns
the hourly series into ONE ROW PER ITEM PER US-EASTERN DAY and merges it into what is stored, so the history only grows:
`item_id, day (YYYY-MM-DD), hours` (usable hours that day: no price spike, listing blip, wipe or long gap, and a sales estimate exists),
`sold_day` (units sold in a 24 hour day, scaled up from the usable hours), `price_day` (median lowest price, gold), `units_day` (median
units listed), `flagged_hours`, `engine_version`. For a day present twice the row with more usable hours wins (the new one on a tie), so
a partial first run is repaired by a later one. A day with under 18 usable hours is stored but not scored. Items covered: those in
engine/items (plus --ids). Run with upload ticked; run again with an earlier `end_date` to backfill. The group test's lifting-together part
needs 14+ complete days per item.

## 2k. Undermine Exchange history (engine/undermine.py, `tools/lake.py undermine` and `undermine_check`; v0.7.16)

Source: https://undermine.exchange/api.html (retail only; regions us/eu/tw/kr; no Forever data). Key in the `UNDERMINE_API_KEY` secret, sent as
`Authorization: ApiKey <key>` with `Accept-Encoding: gzip`. Budget: 3,000 points per sliding hour; a history request costs 3 points compressed.
The client keeps a local count under 2,700 per hour, stops cleanly on HTTP 429 (progress is kept), aborts on 401/403, and never prints the key.

What they give for a commodity: DAILY rows (`day, price, quantity`: price and units listed at the snapshot with the most units that day; back to
2022-09-04 for the items checked, no gaps) and HOURLY rows (`snapshot, price, quantity`: lowest price and total units listed; about 14 days).
They do NOT give sales, listing-by-listing books or Forever. Prices are presumed copper (`undermine_check` measures the ratio to ours).
What it is for: transferable SHAPES (weekday rhythm, how groups move together, how fresh items settle after a patch or launch, holiday and
patch response) as weak priors with wide error bars. Not levels, not sales. Survivorship bias (only items still traded today) and only a few
real launches limit what it can say.

* `undermine` (resumable; tick upload): a balanced sample, at most 30 items per group, spread over era and quality, ordered so any prefix is
  balanced. Default `--batch 800` (2,400 points); run again about an hour later until it prints "All sampled items are done." Writes
  `engine/market_history.parquet` (`item_id, day, price, quantity`) and `engine/market_history_meta.parquet` (`item_id, status ok/empty/not_found,
  fetched_at, first_day, last_day, n_days`) to the PRIVATE dataset only. Items already answered are skipped; network failures are not recorded, so
  they are retried. Prints summary counts only.
* `undermine_check` (about 120 points): about 40 well-measured items (Medium/High confidence, 20+ sold a day, in item_hourly), Undermine hourly rows
  lined up with ours (nearest snapshot within 30 minutes). Reports price ratio (UNITS CHECK, 1.0000 = copper and same feed), units ratio (FEED
  CHECK), and the rank correlation between "units that disappeared from their total this hour" and our sold_est, overall and by how deep the
  stock is compared with hourly sales (tracks well >= 0.6, moderately 0.3-0.6, weakly < 0.3). Summary numbers only.
* Privacy: both modes are excluded from the "Save results" artifact in the workflow; uploads go through a temporary folder, never `out/`.
* Nothing here feeds a forecast yet. Next: a Part 4 of the group test on this long history (weekday rhythm, long-run co-movement, age-aligned
  fresh-item curves).

### 2l. Part 4 of the group test (engine/history.py, `tools/lake.py undermine_test`; v0.7.17)

Reads `engine/market_history` (2k) and `engine/item_groups`; uses no API points; writes no files and prints group-level numbers only (Undermine data stays in the
private dataset). Price and stock are not sales: price = lowest price and stock = units listed at the day's busiest snapshot. Each day is compared with the item's
OWN typical level (the centred 29-day median, lift = log10(value / typical), clipped to +-1), so quantities and slow trends drop out. Only items still traded today are
in the history, so read shapes, not levels.
* 4A weekday rhythm (items with a year of days, 26+ of each weekday): each item's median lift per weekday, re-centred to average 0. An item's pattern is guessed from the
  median pattern of the OTHER items in its group (never itself); vs_flat / vs_all as in part 2; verdict needs 6+ items and a 15% smaller miss, and a swing (best vs worst
  weekday) of at least 3% or it reads "too small to use". stable_r compares the group's pattern in the first and second half of the history; under 0.5 the verdict says
  "(but not stable across years)". Days are as Undermine dates them (probably UTC, so the realm reset may sit a day off).
* 4B moving together: the part 3 test (`comove_by`) on price and stock lifts over years, plus same-day doubling ("spikes"). CONTROL CHANGE (also applies to part 3): the
  "rest of the market" for corr_market is now the MEDIAN over the other groups of each group's mean lift that day (groups with 3+ items, at least 3 other groups), not the
  plain average of every outside item. A few groups with loud shocks of their own used to drag the plain average and make quiet groups look like they "move on their
  own". With fewer than 3 other groups it falls back to the plain outside average.
* 4C fresh items: items that first appear 30+ days after tracking began and have 70+ days of history after that. Price and stock as multiples of the item's own settled
  level (median over days 28-56 of age), median and middle half per age; settle = first age from which the median stays within 15% for 7 days. A week in which 6+
  sampled items (and 2.5x a typical week) first appear is a launch burst; its items are kept apart from the trickle of single new items; the burst weeks are listed so they
  can be checked against real patch / expansion dates, and families with 8+ burst items get their own row. This is the closest retail analogue to Forever's launch.
* Verdicts are screens, not proof: with many small groups a few will look "group-specific" by chance; trust the large groups and the ones that are stable across years.

### 2m. Launch-readiness probes (engine/probe.py, `tools/lake.py api_probe` and `ahledger_check`; v0.7.18)

Neither mode stores anything or writes a file; both print status codes and the SHAPE of answers (field names, types, list lengths, never values), so a job log holds nothing
identifying. They are meant to be run now and again around the Forever launch (Nov 4, 2026).
* `api_probe` (needs the Blizzard credentials the workflow already passes). Builds namespaces as `<static|dynamic>-<name>-us` and asks two cheap paths of each:
  `/data/wow/item-class/index` (static) and `/data/wow/connected-realm/index` (dynamic). Controls: retail, `classic`, `classic1x`, `classicann` must answer 200 (if retail does
  not, the credentials or network are at fault and the report says so). A made-up namespace is the baseline for "Blizzard does not know this" (403 on Oct 26-28). A guess is
  FLAGGED when either status differs from that baseline (a 404 on realms but a 200 on item classes, for example, would mean the namespace exists but has no realms, which is
  what a realmless game might look like). Flagged names, and the known ones for comparison, then get: auctions/commodities, auctions/index, realm/index, market/index, a static
  realm index and item 2589, plus the first connected realm's `auctions` and `auctions/index` when a realm list exists. About 80 requests in all. Candidates are in
  `probe.CANDIDATES` (camelot is in because it is the datamined internal codename; no namespace is called that, but see the Nov 1 note below: the realm names INSIDE the forever namespace start with CAMWOW). Typing words into
  item_ids adds guesses; digits are ignored. This only tests the namespace pattern Blizzard has used for Classic; a Forever API could also arrive as new paths. If a flagged
  namespace answers on auctions, the printed shape is the first look at Forever's listing format and should be checked against checklist items 1, 5 and 8.
* `ahledger_check` asks api.ahledger.com for its own description (`/openapi.json`, `/swagger.json`, `/docs`, `/`, `/v1`, `/api`, `/api/v1`, `/health`), lists every endpoint with its
  parameters and allowed values, then fetches up to 10 GET endpoints that need no parameters and prints their shape. At most 30 requests, one per second, with an honest
  User-Agent. The endpoint paths were NOT known when this was written (the docs could not be opened from the build sandbox), so the mode discovers them; the real puller (markets,
  hourly and daily rows, estimated units sold) is written after its first log is read. Their terms: credit AHledger with a link, do not present their data as our own scan or
  resell it as a feed. Archiving history is not addressed, so ask them before keeping any.


### 2n. First real Part 4 run (Nov 1, 2026; 1,153 sampled items, 1,258,485 item-days, 2022-09-04 to 2026-11-01) and the v0.7.19 changes

What the run said (price and units listed, not sales; survivors only):
* 4C fresh items is the strong result. For the 510 items of the launch-burst weeks, the multiple of the item's own settled price (middle half in brackets): day 0 14.9x (4.0-41.6),
  day 1 6.9x, day 2 5.2x, day 7 2.5x (1.3-4.4), day 14 1.7x (1.1-2.9), day 28 1.5x (1.2-1.8), day 42 0.95x, day 56 0.81x (still drifting down). Units listed, as a share of the
  settled amount: day 0 2%, day 1 9%, day 7 41%, day 14 63%, day 28 91%, day 42 107%. Price settles by about day 38, stock by about day 24. Families differ: gear and gems keep
  a premium longer (day-7 price 3.9x and 3.5x) than materials (1.9x) and consumables (1.7x). The 47 single new items spike less (day 0 5.7x) and settle sooner.
  The burst weeks were Dragonflight launch (2022-11-28), The War Within early access (week of 2024-08-19; early access Aug 22, launch Aug 26) and Midnight early access
  (week of 2026-02-23; early access Feb 26, launch Mar 2), plus smaller patch weeks (2023-05-01 is Patch 10.1). The first day an item exists is its early-access day.
* 4A weekday rhythm: stock peaks Tuesday and is lowest Thursday or Sunday in almost every group, stable across years (stable_r 0.92 overall) but only about 1% overall; gear
  slots show 5-11% (Cloak 5.0%, Feet 10.9%, error cut by about 30%). The PRICE rows were unmeasurable (swing 0.00, vs_flat NaN): for most items the median day's price sits exactly
  at the item's normal level, so a median-based weekday profile is zero everywhere. Not evidence of "no rhythm".
* 4B moving together (day by day): price correlation inside a group is about 0.06, no higher than against the market. Shared moves exist in a few small groups (class glyphs, gear
  Neck, Miscellaneous Reagent: stock correlation 0.46 vs 0.09) and glyph / gear groups spike together (2-3x as likely as normal); with ~40 groups tested, expect some by chance.

Changes in v0.7.19 (`engine/history.py`, `undermine_test`): (1) PRICE is shown twice, median day and average of all days (`price_mean` in `rhythm`), plus `flat` = how often each
family's price is within 0.5% of its normal level; (2) `comove_weekly` repeats the moving-together test on weekly averages (a week needs 4 days; no spikes); (3) fresh-item curves run
to 112 days (ages 70, 84, 112 added) to see where the price finally lands; (4) headings no longer start with "==" (a spreadsheet turns such a line into #ERROR!).

### 2o. Launch forecaster (engine/launch.py, `tools/lake.py launch_backtest` and `launch_now`; v0.7.19)

Turns 4C into a forecast. For an item seen on day k at price P (and P0 on its first day), the settled price (median over days 28-56 of age) is P / m, where m is the multiple items
had on day k. Ranges come from the spread of m among other items. Four guesses are scored on launches that were HIDDEN from the fit (leave one launch out; a launch = burst weeks
within 14 days of each other, 15+ items to test on, 30+ in the others): naive (settled = today's price), age (all other launches' items), family (same family, 15+ items) and path
(age plus the item's own fall since day 0: a + b * log10(P_k / P_0)). Reported per day 1, 2, 3, 5, 7, 10, 14, 21: the typical miss as a multiple, how often the middle-50% and
middle-80% ranges held the truth (should be near 0.5 and 0.8), and the width of the 80% range. `best_model` per metric and day is the smallest miss (the plain age guess unless another
is 3% better). `launch_backtest` with upload stores `engine/launch_prior` (scope all + families with 15+ items; metric price / stock; model age / path; log10 quantiles l10..l90, line a, b).
`settled_estimate()` turns a prior row and an observed price into mid, 50% and 80% ranges (None past 24 days). `launch_now` applies the prior to items young in `engine/item_daily`
(from `daily`; ages count from each item's first day in that table, so the table must start at the launch).
Limits: retail launches add items to an economy that already has gold and stock; Forever starts empty, so levels may differ. The ranges are a prior; checklist item 11 says how they are
scored on Forever's own data. Tested on synthetic launches only (hit rates near 0.5 / 0.8 on hidden launches); the first real result is the `launch_backtest` log.


### 2p. Forecast bake-off (engine/forecast.py, `tools/lake.py forecast_test`; v0.7.22)

Question: for SETTLED items (56+ days past first appearing; the launch forecaster owns the weeks before), which simple method beats "tomorrow equals today", by how much, and are its
error bars honest? Daily price (gold, lowest price) and units listed from `market_history`, in log10, items with 120+ observed days. Horizons 1, 3, 7 days. Walk-forward: the dated span
after its first 56 days is cut so testing starts 40% of the way in and runs in 3 stretches; for each stretch the last 30% of the days before it calibrate the ranges and the days before
that tune the pooled models; every origin is scored (every 3rd day) with only days up to the origin. All methods are scored on exactly the same cases.
Methods: naive; week_ago (same weekday last week); median7; ema50 / ema25 / ema10 (alpha .5 / .25 / .1); weekday (level + learned weekday pattern, the plain-arithmetic cousin of SARIMA); sarima (engine/sarima.py: SARIMA(p,d,q)(P,D,Q)7 per item, parameters
fitted on the days before the calibration stretch and then held fixed while it is filtered forward, best of four specs by AIC: (1,0,1)(1,0,1) around a fixed level, (0,1,1)(0,0,1) local level,
(0,1,1)(0,1,1) airline, (1,1,1)(1,0,1) damped trend; short gaps interpolated, a failed fit falls back to ema25 and is counted; needs scipy, added to the workflow's `pip install` line, nothing in
requirements.txt touched); pooled (one Huber regression across all items: deviation of today from the item's EMA25, EMA50-EMA10, median7-EMA25, weekday of
the target); kinked (pooled with separate slopes for ordinary moves and for spikes beyond 26% from the EMA25, so it can stay put for small moves yet pull back after a spike); pooled_grp (pooled + how far the item's group peers sit from their own EMA25 today: the cheap ablation of whether neighbours help, standing in for a graph model).
Ranges: the method's own signed misses from the calibration days, quantiles q10 / q25 / q75 / q90, kept apart for five levels of recent jumpiness (14-day spread of daily changes), so calm
items get narrower ranges. Measures per method and horizon: miss_pct (median miss as % of price), avg_pct, vs_naive (mean |miss| / naive's, below 1 beats it; this is the "MASE" of the
notes without the in-sample scaling), wape (pooled, pricey items weigh most) and wape_med, bias_pct and track (signed; positive = real price above the guess), rmse_pct, p95_pct, big_pct
(share over 25%), dir_hit (only when the guess moves 3%+ from today) and dir_n, hit50 / hit80 / width80_pct. `best_models`: naive unless another is 3%+ better on average |miss|. Also printed: best guess by family, results by item jumpiness (five levels set on each fold's
calibration days: daily swing, naive's typical and average miss, best method, vs_naive, hit80, width80), and a SARIMA CHECK that rescored every method only on the item-fits where SARIMA
really fitted (an item needs 120 days before the stretch; the rest fall back to ema25).
Limits: retail survivors only (items still traded today), daily data not hourly, price is the lowest price (undercut noise), no sales; tested on synthetic worlds (finds a planted weekday
rhythm, finds a planted shared shock, loses to naive on a random walk, ranges near 0.5 / 0.8 on stationary noise); the first real result is the `forecast_test` log. SARIMA is our own
conditional-least-squares implementation, not statsmodels (the build sandbox cannot install it, so code calling it could not be tested): it recovers planted AR / seasonal / MA parameters,
matches the textbook AR(1) and MA(1) forecasts, reproduces an exponential average with (0,1,1), and never reads days after the origin. If SARIMA wins on the real log, an exact-likelihood
statsmodels fit can be added as one more candidate and compared on the same cases. AIC is used only to pick among the four SARIMA specs; Ljung-Box is not computed: the hidden-days test is the stricter check. Next (after reading the log): a daily forecast LEDGER that stores each day's forecasts with ranges and grades them when the
actual day arrives (prices from our own scans, sales from Journalator), so ranges are re-calibrated on Forever's own misses.

First real `forecast_test` log (Nov 2, 2026; 1,137 items, tests 2024-06-07 to 2026-11-01), read before v0.7.24:
* The target is jumpy: naive's 1-day typical miss is 6% but its average is 28% and 28% of days miss by over 25% (prices sit, then jump). Average miss against naive (vs_naive): price 1 day
  best 1.00 (naive's typical miss still wins), 3 days 0.92 (pooled), 7 days 0.88 (pooled; EMA / weekday 0.89); stock 0.97 / 0.87 / 0.87 (pooled). Crafting materials gain most (0.86-0.87 at 3-7 days).
* SARIMA (our CSS fit) lost: 1.15 / 1.00 / 0.96 against a plain EMA's 1.20 / 0.94 / 0.89 (price). 1,108 of 3,411 item-folds (32%) fell back to ema25 (under 120 days of history before the
  stretch). weekday == ema25 (0.95 vs 0.94 at 3 days; 0.89 vs 0.89 at 7), so the weekly rhythm SARIMA exists to catch is not in daily lowest-price series. pooled_grp == pooled to two decimals:
  group peers add nothing at daily resolution (recipe edges not tested). Ranges honest: hit50 ~0.50, hit80 0.79-0.80 (SARIMA 0.77-0.78); wide: 7-day 80% width 180-240% of price.
* Open: robust price target (the lowest price is dominated by single cheap listings; our own hourly data can define a depth-based price), pooled models work from day 56 while per-item
  models cannot (a Forever point for the first weeks), and v0.7.24 adds the kinked model and the by-jumpiness / SARIMA-fitted-only tables to see where forecasts are tight enough to trade on.

Second `forecast_test` log (v0.7.24, Nov 2, 2026) and v0.7.25:
* SARIMA is out as a default: on the clean subset where it actually fitted (216,749 one-day cases) it scored 1.15 / 1.01 / 0.97 against naive at 1 / 3 / 7 days, plain EMAs 0.88-0.89 at 7 days,
  and its 80% ranges ran narrow (0.75-0.78). `forecast_test` now leaves it out unless the word `sarima` is in item_ids (it was 2.5 of about 3 minutes of run time).
* kinked is only marginally better than pooled (0.91 / 0.87 at 3 / 7 days vs 0.92 / 0.88) and does not recover naive's 1-day typical miss (9.2% vs 6.0%). pooled_grp == pooled again.
* By jumpiness (price): calmest fifth (typical daily swing 7%): naive best at 1, 3, 7 days; average miss 8.6% / 16% / 24%, 80% range 21% / 57% / 93% wide. Jumpiest fifth (daily swing 137%):
  average miss 60% / 104% / 135% for naive, 80% range 375% / 532% / 674% wide; models reach 0.85 / 0.80 of naive at 3 / 7 days. Models help only the jumpier three fifths at 3-7 days. hit80 is
  0.78-0.81 at every level. About 40% of cases are in levels 4-5 (daily swings of 60%+): the lowest-price target itself is the main limit. Stock improves at every level (0.84-0.93 at 3-7 days).
* v0.7.25 adds `target_test` (engine/target.py): our `item_daily.price_day` (median of the day's usable hourly lowest prices, spike hours dropped, 18+ usable hours) against Undermine's
  (lowest price at the busiest snapshot) on shared items and consecutive days: size of daily moves, share over 25%, correlation, and how many of Undermine's 25%+ moves ours also makes
  (confirmed) versus barely registers (blip). Day labels are matched as written (ours US Eastern, theirs probably UTC). Needs item_daily (mode daily, upload ticked) and market_history.

First real `target_test` log (v0.7.25, Nov 2, 2026; only 3 shared days, 2,867 paired changes, 1,042 items) and v0.7.26:
* Our `item_daily.price_day` (median of 18+ hourly lowest prices, spike hours dropped) is NOT much calmer than Undermine's single-snapshot lowest price: average daily move 21.8% vs 24.2%
  (1.1x), share of moves over 25% 21% vs 25%, corr of daily changes 0.49, 77% of item-days within 10% on level (median gap 0.0), of Undermine's 25%+ moves 48% confirmed by ours and 25%
  barely registered. So the day-to-day volatility of the lowest price is mostly real (thin books, undercut cycles), and the forecast limits found in `forecast_test` apply to our own table
  too unless the price definition changes. Caveats: 3 days (about 2 changes per item), different hours (a busiest snapshot vs the whole day) and day labels (ours US Eastern, theirs probably UTC).
* v0.7.26: `item_daily` gains `price_p15` / `price_p30` = the median over the day's usable hours of the price at which the cheapest 15% / 30% of the UNITS listed have been reached (walk up the
  book; engine/daily.py `depth_prices`; vectorised, no per-item loop). One cheap listing sets the floor but cannot set these. Rows stored before v0.7.26 have them empty; rerunning `daily` over
  a day fills them (a tie on usable hours goes to the new row). `target_test` now first compares our own definitions on the same item-days (changes, avg / typ move, share over 25%, jump,
  calmer_x against the floor, how far above the floor each sits), then the Undermine comparison. If a depth price is clearly calmer it becomes the forecast target (TSM's AuctionDB market value is
  the same idea; the addon export's MarketPrice column is the Forever-side equivalent when only scans are available).

### 2q. Hourly forecast bake-off (engine/forecast.py `Cfg` / `HOURLY`, `tools/lake.py forecast_hourly`; v0.7.27)
Same test as 2p, run on OUR OWN hourly snapshots instead of Undermine's daily history. Reads the cached snapshots like mode `daily` (items = engine/items plus item_ids; set `days` to about 14),
writes nothing, prints group-level numbers only. Why: the daily test could not see a 24-hour cycle, and trades happen on an hours horizon.
* Data: `hourly_frame` takes the LAST snapshot in each US Eastern clock hour (the same clock the daily table uses; the repeated hour when clocks go back is merged into one slot, so the cycle
  position stays continuous). Price = lowest unit price (gold), stock = units listed. A snapshot with under half the median number of listings is a partial capture (Oct 26 had 13K-31K against
  262K) and counts as missing; they are listed in the output. An hour with no snapshot is a NaN gap; items with under 72 readings are left out (so the sample leans to items that are almost always listed).
* Settings (`Cfg`, defaults = the daily test, so daily results are unchanged): season 24, horizons 1 / 4 / 12 / 24 hours, 48 hours of burn-in before an item is scored, rolling median of 6,
  jumpiness = spread of hourly changes over 24 hours, every hour is an origin (step 1), 3 folds starting 40% in. Thresholds: a big miss is over 10% and a direction call counts at 1%+.
  Candidates: naive, day_ago (same clock hour yesterday), median6, ema50 / ema25 / ema10 / ema04 (ema04 remembers about a day), hour_of_day (level plus a learned 24-slot pattern, the Holt-Winters
  analogue), pooled / kinked / pooled_grp (as before, with 23 target-hour dummies and one extra feature: the fast average against the ema04 level, so the fit can learn a pull back to the daily level).
  Display names replace the weekly ones (week_ago -> day_ago, median7 -> median6, weekday -> hour_of_day); the engine keeps the internal keys. At 24 hours day_ago equals naive exactly.
  SARIMA is not run: it needs several cycles per item and only days of hours exist (the 24-hour version can be added when weeks have accumulated).
* New guard (also protects the daily test): if a pooled model has too few rows (under 2,000) to fit in a stretch, that stretch is skipped for EVERY method, so all methods stay scored on identical cases.
* "IS THERE A DAILY CYCLE?" (`hour_profile`): per clock hour, how far the typical item sits from its own centred 24-hour average (median over items and days) and the share of items leaning the same
  way (about 0.50 at every hour = no shared cycle, near 1.00 = items move together). Independent of any model.
* Honest limits: about a week of usable hourly history (first snapshots Oct 25 19:31 ET, Oct 26 missing), so the pooled fit's first stretch is thin at 24 hours and every figure is PRELIMINARY;
  retail is the test bed, and Forever's own hourly track record has to be earned the same way.

First real `forecast_hourly` log (v0.7.27, Nov 2, 2026; 2,998 items, 187 clock hours Oct 25 19:00 to Nov 2 13:00 ET, of which the loader had dropped Oct 25 20:00 to Oct 26 18:00 as partial captures, so about 6.75 continuous days):
* Price: "it stays put" is unbeaten at 1 and 4 hours (best rival: pooled 1.08 and 1.01 of naive's average miss). Typical 1-hour miss of naive is 0.00% (more than half of item-hours do not change), average 8.5%, 12.8% of
  cases miss by over 10%; WAPE 11.7%. At 12 and 24 hours kinked / pooled are 0.96 and 0.93 of naive; gains sit in crafting materials (0.91, 0.89), enhancements and the jumpiest fifth of cases (0.91 at 12h, 0.84 at 24h);
  consumables and gems get nothing. Same shape as the daily test: pullback after a spike, nothing else.
* No shared daily price rhythm: the typical item sits 0.00-0.03% from its 24-hour average at every hour, items leaning the same way 0.52-0.58; hour_of_day loses at 1-12 hours (1.65, 1.14, 1.00) and ties at 24 (0.96).
* Stock (units listed) has a small shared daily cycle: lowest 6-9 AM ET, highest 8-10 PM ET, about 5 points peak to trough, 72% of items leaning the same way at 10 PM. Too small to forecast alone (1-hour naive
  miss 12.5%); pooled / kinked are 0.94 / 0.93 of naive at 12 / 24 hours, nothing at 1-4 hours.
* Ranges hold: hit80 0.79-0.82 for every method and horizon (hit50 0.69 for naive at 1 hour because the median change is exactly zero). The calmest fifth of cases has a 1-hour 80% range under 1% wide and holds 83-89%.
  "Calm over the last day" does not carry over a day: the calmest fifth at 24 hours still missed 44% on average (hit80 0.77).
* Caveats found in the log: (1) the 48-hour burn-in was counted in clock hours, so the Oct 26 hole ate half of it and hour_of_day / ema04 had about one day to learn; (2) day_ago's median miss is exactly 10.00% and its 95th
  percentile exactly 400.00%, which points at price steps in the cheapest items (copper granularity), not yet confirmed.
* v0.7.28: burn-in for hourly is now counted in real readings an item has had (`Cfg.age_by_obs`; testing starts once half of the items are burned in); both bake-offs print a BEST GUESS BY PRICE LEVEL table (under 1g /
  1g to 10g / 10g to 100g / 100g and up, by each item's median price) so the result can be checked on items worth trading.

First real `forecast_hourly` price-level split (v0.7.28 log, Nov 2, 2026; same 2,998 items): the average miss of "stays put" falls steeply with price: 1 hour 20.2% (under 1g) / 11.8% (1-10g) / 7.1% (10-100g) /
3.7% (100g and up); 4 hours 43 / 26 / 17 / 8.8%; 12 hours 73 / 45 / 30 / 17%; 24 hours 95 / 62 / 41 / 25%. Nothing beats it at any horizon for 100g and up; 10-100g gains 5% only at 24 hours; the 4-11% pullback gains
at 12-24 hours sit in items under 10g (so the earlier crafting-materials gain is mostly a cheap-item story). Stock gains 4-8% at 12-24 hours in every price tier. The burn-in fix changed nothing (hour_of_day 1.65 / 1.14 /
1.00 / 0.96 again), so there is no daily price rhythm in the floors. hit80 stays 0.79-0.82. The "exactly 10.00%" worry was a coincidence (10.29% now). Reading: for items worth trading the forecast is today's price
plus an honest range; the edge is in range width and sell speed, not in predicting direction.

### 2r. Forecast ledger (engine/ledger.py, `tools/lake.py forecast_ledger`; v0.7.29)
The live record the bake-off cannot give: guesses are written down with ranges BEFORE the day happens and graded against our own daily price (`item_daily.price_day`, complete days only) when it arrives. Run it
after mode `daily`, once a day, with upload ticked. Reads item_daily, market_history (Undermine) and forecast_ledger; writes only engine/forecast_ledger.parquet in the private dataset; prints group-level numbers only.
* Table `forecast_ledger` (one row per origin day x item x horizon x method; never rewritten except `actual` being filled in): made_day (the origin: latest day on which at least half as many items are complete as
  on the best-covered day), item_id, horizon (1, 3 or 7 days), method, guess (gold), lo50 / hi50 / lo80 / hi80 (gold), origin (price on the origin day), jump_bin (1 calmest to 5 jumpiest at the time), target_day,
  actual (gold, empty until the target day has 18+ usable hours), made_at (UTC), engine_version.
* Methods side by side: naive (stays put), ema25 (exponential average, weight 0.25), kinked (the bake-off's pooled regression with separate slopes for small moves and spikes, fitted on the Undermine history before its last
  180 days and applied to OUR series: the test of whether retail-trained coefficients transfer). Eligibility: observed days in the last 14: naive 5, ema25 7, kinked 12 (and 4 day-to-day changes for the jumpiness level).
* Ranges: for each (method, horizon) the q10 / q25 / q75 / q90 of that method's misses over the last 180 days of the Undermine history, kept apart for five jumpiness levels (edges from the same window). A PRIOR: Undermine's
  price is not exactly our price (ours is a median of hourly floors and was about 1.1x calmer), so the scoreboard's hit50 / hit80 on our own series are the check. Recalibrating from our own graded misses is the next step once
  a few weeks of grades exist.
* Honesty rules (all tested): only days up to the origin are used (later days are cut off before any model sees them); a row is not written when the item's target day is already complete; a graded row is never changed;
  a second run on the same origin day writes nothing; the target day must be complete (18+ usable hours) to grade.
* Output: a status line, the prior 80% range widths for naive, rows written, a SCOREBOARD per horizon and method (cases, items, miss_pct, avg_pct, vs_naive on exactly the same cases, bias_pct, hit50, hit80, width80_pct), the
  same by price level (under 1g / 1g-10g / 10g-100g / 100g and up, from the origin price), and what is still waiting for its day.
* Timeline on retail: first guesses today (naive for items with 5+ observed days), ema25 from about Nov 3, kinked from about Nov 7; +1 day grades the next day, +3 after three, +7 after a week. On Forever the same table starts
  at launch: items younger than 24 days belong to the launch forecaster (engine/launch.py), and this ledger takes over per item once it has the days.

### 2s. Daily schedule (v0.7.30)
* `.github/workflows/daily_chain.yml` (identical copy in `tools/`) runs mode `daily` (default `--days 4`, so a run also repairs up to three missed ones) and then mode `forecast_ledger`, both with `--upload --store hf`.
  Manual start only (`workflow_dispatch`, inputs `days` and `end_date`); concurrency group `daily-chain`, never two at once; step 2 does not run if step 1 fails. Needs only the `HF_TOKEN` secret.
* Start time: 06:17 UTC every day. The US Eastern day ends at 05:00 UTC (04:00 UTC in summer time), its last snapshot is published about :31 past the hour and fetched about :01 after, so the day is on the
  Hugging Face mirror by about 05:05 UTC; 06:17 leaves an hour of slack and avoids the top of the hour, when scheduled jobs queue up. 06:17 UTC is 1:17 AM Eastern now and 2:17 AM after the March change.
* The trigger is the owner's existing Cloudflare Worker (the one that starts `compact.yml`; code lives in the Cloudflare dashboard, not in this repo). It reads `event.cron`: the cron "17 6 * * *" starts
  `daily_chain.yml`, any other cron starts `compact.yml` as before. Both calls are `POST /repos/{owner}/{repo}/actions/workflows/{file}/dispatches` with `{"ref": "main"}` (answer 204), using
  the Worker's existing secret `GITHUB_PAT` (no second token). The second cron time is added under the Worker's Settings > Triggers > Cron Triggers.
  If "Daily chain" does not appear in Actions at 06:17 UTC but compaction runs then, the cron text entered in the dashboard does not match the text in the Worker exactly.

### 2t. Forecasts in the front end (v0.7.31; `app.py`, `lib.py`)
* Reads `engine/forecast_ledger.parquet` like the other tables (private dataset, `HF_TOKEN` secret). A missing or empty ledger is fine: the Forecasts tab says none are recorded yet and the item view shows nothing extra.
* Item view (Market / Flips tabs, or the search box): a selector "Forecast drawn on the chart" (methods that have forecasts for the item; default "Stays put", the bar everything must clear) adds to the hourly price chart
  (a) diamonds for the item's daily prices known to the ledger (`origin` on each guess day, `actual` on each graded target day, actual wins), (b) a fan from the latest guess day's price through the
  1 / 3 / 7 day guesses with the 80% (light) and 50% (dark) ranges, and a dotted guess line. A daily price is drawn at midday US Eastern of its day (`lib.day_to_utc`). Below the chart, an expander lists every forecast
  for the item with the real price and "inside 50% range / inside 80% range / outside: higher / outside: lower / waiting". An item with no forecast yet gets a one-line reason (days of own prices needed per method).
* Forecasts tab: counts (recorded, items, graded, waiting); the scoreboard per horizon and method (typical and average miss, "closer than staying put?" = vs_naive, share inside the 50% / 80% ranges, typical real-vs-guess
  lean, typical 80% width, and "first look" below `lib.MIN_TRUST_CASES` = 100 cases); bar charts of the inside-50 / inside-80 shares against their targets; running totals by target day for a chosen horizon (inside 80% and
  vs_naive); an expander with the same scoreboard by price level. Before anything is graded it shows what is waiting and when the first grades arrive.
* `lib.ledger_scoreboard` repeats `engine.ledger.scoreboard` so the front end needs no engine import; a test compares the two on a made-up ledger. Plain pandas only, as the rest of `lib.py`.
* Size: the app downloads the whole ledger (cached 10 minutes). Rows grow by about (items forecast x 9) per day; if that ever becomes slow, the ledger gets split by year or the page reads only recent rows.

### 2u. Forever fetcher (v0.7.32; `fetch_forever.py`, `.github/workflows/forever_fetch.yml`, copy in `tools/`)
* A separate script; the retail `fetch.py` is not changed and not imported. Its frame code is a copy of the retail `build_frame` / `variant_key` (a test compares the two when `fetch.py` can be imported; checked by hand against the live
  file on Nov 3: identical frames and variant keys), so the Forever lake reads like the retail lake: `forever_<connected realm id>/year=/month=/day=/hour_HH_MM.parquet`, one row per auction, whole `item` struct kept.
* Each run: realm list (`dynamic-forever-us`) -> for each wanted realm id its realm record -> `connected_realm.href` -> connected realm id -> `.../connected-realm/{cid}/auctions` with If-Modified-Since. Default realms: the eight US PvP
  game realms of the Nov 3 probe (ids 6798, 6922, 6924, 6926 = US1 Realm 2/4/6/8; 6800, 6930, 6932, 6934 = US2 Realm 2/4/6/8). Override with the workflow input `realm_ids`, the repository variable `FOREVER_REALM_IDS` (v0.7.33; input wins) or env `FOREVER_REALM_IDS`; if the realm list is live and none of the wanted ids is in it the run alerts and prints the list; the auction path with `FOREVER_AUCTION_URL`
  (use it if Blizzard serves listings somewhere else). Realms are shards of one ruleset if the Nov 3 probe's one-connected-realm-per-realm reading is wrong, otherwise eight separate markets: the first week of data says which.
* Storage: Hugging Face only, a separate private dataset (default `wow-forever-data-lake`, constant `HF_REPO_ID`; created on the first commit if the token may create repositories). No Backblaze, so no new cap pressure and no new key. Each live run is ONE
  commit holding the new snapshots plus `system/forever_markers.json` (last Last-Modified per source) and `system/forever_sources.json` (source -> realm id, name, connected realm id). Markers travel with the data, so a failed commit advances nothing.
* Dark is not a failure: 403 / 404 from Blizzard (realm list, realm record, auctions) means "not served yet" and the run ends green with a one-line summary. The first time a source returns auctions, one Discord message says Forever is live
  (needs the `DISCORD_WEBHOOK_URL` secret). A 200 with an empty auction list is logged and kept out. 429 / 5xx are retried three times, then count as failures (red run + alert); a realm that fails does not stop the others.
* Switch: the job runs after every run of "Hourly Auction Extractor" (`workflow_run`) only while the repository variable `FOREVER_ENABLED` is `true`; a hand-started run always goes. Secrets: BLIZZARD_CLIENT_ID / SECRET, HF_TOKEN (write access to the
  Forever dataset), optional DISCORD_WEBHOOK_URL. Error text has the dataset name blanked out (`clean()`), so no account name reaches a job log.
* Not yet Forever-aware: the engine (`tools/lake.py` reads source `commodities`; per-auction `kind="auction"` classification is checklist item 5) and the front end. They wait for the first real files; the lake name will become a setting then.

Fourth `api_probe` log (Nov 4, 2026, about 5 PM ET, an hour before the expected go-live): CHANGES. Forever STATIC data is now served: `static /data/wow/item/2589` and the item classes went 403 -> 200 (the first flip on the launch-day
watch list). The DYNAMIC realm index went backwards: 404 in US, EU and TW (it was 200 with 51 CAMWOW test realms on Nov 3), connected realms 404, commodities / auction-house index / listings unreachable. Reading (a guess, unverified): Blizzard cleared the
staging realm list while setting up the real launch realms, so the preloaded realm ids (6798 ...) may not exist after launch. Consequences: (1) v0.7.33 makes the realm ids settable without a push (repository variable `FOREVER_REALM_IDS`, or the workflow
input) and alerts, with the realm list in the message, when the list is live but none of the wanted ids is in it; (2) Forever item names / classes can now be read from the static namespace (an items table for Forever is possible before listings exist).

First real `api_probe` log (Nov 1, 2026, 21:4x ET), and what v0.7.20 adds to the mode:
* Known namespaces behaved as expected (retail, classic, classic1x, classicann all 200); the made-up namespace answered 403 on both paths. Of 30 guesses exactly one differed: `forever`
  (`static-forever-us` item classes 403, `dynamic-forever-us` connected-realm index 404). `camelot` and every other guess matched the made-up one.
* Followed up, `dynamic-forever-us` `/data/wow/realm/index` answered 200 with 51 realm entries (fields key, name, id, slug). Commodities, `auctions/index` and `market/index` answered 404;
  static realm index and item 2589 under `static-forever-us` answered 403 (static data not published). Compare: classic 50 realms and full listings, classic1x 66 realms with an auction-house
  index of 3 houses but no listing path, classicann 5 realms, same. So Blizzard has created a Forever dynamic namespace with a realm list; whether auction data will be served, and under
  which path, is not yet visible. Era-style (index only, no listings) is a live possibility and is what the fallbacks in checklist item 9 are for.
* 51 realm entries sits oddly with "no realms, 18 markets" (6 US markets would be expected); the entry names will say what they are.
* v0.7.20: for any flagged namespace the probe now (1) prints the realm names and ids, (2) follows the first three realms through `/data/wow/realm/{slug}`, the connected realm it names,
  `auctions/index`, `auctions` and the first auction house's listing, (3) asks the eu, kr and tw hosts the same realm-index question, (4) prints a plain VERDICT (namespace known, realms,
  static data, auction index reachable, auction LISTINGS reachable). `classic` and `classic1x` get the same follow-up for comparison. Realm names are game data, not account data, so they are printed.

Second `api_probe` log (Nov 1, 2026, ~21:55 ET) and v0.7.21:
* Same statuses as the first run (still only `forever` flagged; realm index 200, connected-realm records 404, no auction path reachable). New: the 51 realm names, all starting "CAMWOW" (almost
  certainly Camelot WoW; the earlier "no Camelot in the API" answer had only looked at namespace names). 25 look like game servers ("US1 Realm 1 PvE" .. "Realm 10 RP": clusters US1 10, US2 11,
  AU1 4; rulesets PvE 9, PvP 9, RP PvP 4, RP 3); 26 look like internal infrastructure (US2 CSI 80-88 and 180-188 = 18, Web x3, GMSS x3, Account, Log). That looks like a staging / test realm list, not the
  launch market list. EU answers with 26 realms, Taiwan 5, Korea 403.
* Hypothesis (from beta play: one US PvP server, but a friend showed on "PvP 2" and the user on "PvP"): the realms are SHARDS of one logical server per ruleset, all sharing one market. Test:
  each realm record names its connected realm (`connected_realm.href`); in Classic a connected realm is what shares one auction house. v0.7.21 reads every realm record and prints how many
  different connected realms the realms point to, with the ruleset mix of each. Few connected realms for many realms supports the shard idea and tells us how many separate markets (sources) to
  expect; one connected realm per realm says the list is not grouped (yet). The connected-realm records themselves (`/data/wow/connected-realm/{id}`) still answered 404 on Nov 1.
* Also printed since v0.7.21: the realm names of the EU / KR / TW namespaces that answer.

Third `api_probe` log (Nov 3, 2026, afternoon ET): no change in what Blizzard serves for Forever. `forever` is still the only flagged namespace (item classes 403, connected realms 404); realm index 200 with the same 51 CAMWOW entries
(Normal 35, PvP 9, PvP RP 4, Roleplaying 3); static item 2589 403; commodities, auction-house index and every connected-realm path 404; realm records for the first three realms answer 200 but their connected-realm records 404.
* New: the realm map (first run with v0.7.21's mapping): 51 realm records point to 51 DIFFERENT connected realms, so nothing in the list says realms share an auction house. That is no support for the shard idea (one market
  per ruleset), but weak evidence either way: the connected-realm records themselves are still 404 and the list looks like a staging list, so the mapping may be placeholders.
* New: Korea answers 404 on the realm index where it answered 403 on Nov 1 (403 = Blizzard does not know the namespace, 404 = it does and finds nothing there): the Korea namespace appears to have been created since. EU 26 realms, TW 5, unchanged.
* To watch on launch day (Nov 4): static `item/2589` going 403 -> 200 (item data published), then `connected-realm/{id}` and `connected-realm/{id}/auctions` going 404 -> 200 (listings). Run `api_probe` at launch and a few times through the day.
