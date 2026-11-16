v0.7.40: the buy-below prices now run by themselves every day and show up in the app on a new "Buy targets" tab. This REPLACES v0.7.39 (it contains everything in it), so if you have not committed v0.7.39, skip it and use this zip only. If you already did, this zip just overwrites those files.

What it does not touch: the retail fetcher, the Forever fetcher, the hourly lake files, the forecasts, the other app tabs, and requirements.txt. Nothing here buys or sells anything.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py and app.py). Overwrite when asked. Do NOT create a frontend folder.
  app.py                                 replaces the existing file (new "Buy targets" tab)
  lib.py                                 replaces the existing file (helpers for that tab)
  engine/targets.py                      NEW (if you skipped v0.7.39) or replaces it
  engine/windows.py                      replaces the existing file
  engine/dips.py                         replaces the existing file
  tools/buy_windows.py                   replaces the existing file (now has --only targets and --upload)
  tools/buy_windows.yml                  replaces the existing file (copy of the manual workflow)
  tools/daily_chain.yml                  replaces the existing file (copy of the Daily chain, now with a third step)
  .github/workflows/buy_windows.yml      replaces the existing file
  .github/workflows/daily_chain.yml      replaces the existing file
  tests/test_targets.py                  NEW or replaces it
  tests/test_windows.py                  replaces the existing file
  tests/test_dips.py                     replaces the existing file
  tests/test_daily_chain.py              replaces the existing file
  tests/test_frontend_lib.py             replaces the existing file
  tests/test_app_smoke.py                replaces the existing file
  docs/DATA_CONTRACT.md                  replaces the existing file
requirements.txt and every other workflow are NOT in this zip and do not change.

AFTER YOU PUSH (three steps, then it runs on its own)
  1. Actions > Daily chain > Run workflow, defaults. It now ends with a third step, "Buy targets". The first run takes a few minutes longer than before. When it finishes, the run's summary shows the report for that step, and the private dataset gets four new files in engine/:
       buy_targets.parquet  buy_calibration.parquet  buy_when.parquet  buy_meta.parquet
     If that third step fails, send me its log. (Nothing else is affected: the daily history and the forecast ledger are already saved by the two steps before it.)
  2. Open the app, wait up to 10 minutes or use the menu (top right) > Clear cache, and click the new "Buy targets" tab.
  3. Nothing more. Your existing daily schedule (06:17 UTC, the Cloudflare Worker) already starts the Daily chain, so the list refreshes every day by itself. The tab shows when it was last built and turns yellow if it is older than a day and a half.

WHAT THE TAB SHOWS
  A shopping list for gathering inventory. For every item, the price at or under which it is a good buy against its own last 7 days: pick "Good" (bottom 25% of its hours), "Great" (bottom 10%) or "Steal" (bottom 3%).
  - Chance in 1 / 3 / 7 days: how often a target like that was actually reached on days the tool had not seen when it set the price (looked up by price level and how deep the target is).
  - Units near floor: how many units were usually listed close to the floor when the price was that low.
  - Worth (g): units near the floor x how far the target is under the 7-day median. It is what you save against paying the median, not a profit. The table is sorted by it.
  - Filters: price level, item group, name or ID, minimum chance within 3 days (starts at 50%), minimum depth under the median (starts at 3%), minimum units. Click a row to see that item's charts (only for items that are in the Market table).
  - "Copy the list for your in-game notes" gives one line per item, "Saronite Ore: buy below 12g 34s 56c", best first, and a CSV download.
  - "How reliable are these prices?" shows how often each depth of target was reached, how long the wait was and the saving. A shallow target is reached nearly always and saves almost nothing; a deep one is the real bargain and is reached less often.
  - "When is it cheapest?" shows the hour-of-day chart with a plain verdict. The verdict uses days the pattern was not learned from. Expect "no reliable best hour" for most pools: the earlier tests found the clock worth about 0 to 2%.

WHAT IT DOES NOT CLAIM
  - It never says a price is on the market now. The list is built from hourly prices up to about half a day before the newest snapshot, and refreshed once a day. In game, the scan decides whether the price is there.
  - It says nothing about what you will do with the items.
  - After a real shift (a patch, a new season) the targets lag by about a week because they come from the last 7 days.

THE BUY WINDOWS WORKFLOW STAYS as the manual testing tool (clock test, weekday test, dip test, the full report). Run it whenever you want a fresh test report; it no longer needs to be run for the app.

ONE THING TO KNOW: the four new tables go to the PRIVATE dataset only. The Daily chain's log now also shows the buy-below report (item names, ids and prices from the public auction house; no account or character information). The manual Buy windows workflow still uploads an artifact; on a public repo any signed-in GitHub user can download it.
