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

## 9. Next steps (in order)

1. Loader: Parquet lake (B2/HF via DuckDB) → `Snapshot` list.
2. Choose experiment items from your own v2 data (liquid enough to sell, cheap enough to risk).
3. Buy/list experiment protocol and log template. Real sales are the ground truth we lack.
4. Fit `cancel_share` and the blind-spot correction from that log; re-run validation on real data.
5. Verify `time_left` buckets.
6. Crafting margins, nightly derived job, Discord alerts.
