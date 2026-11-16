v0.7.37: the first half of the buy-window work. (1) The daily run now keeps a permanent HOURLY price history (the old hourly table keeps only 8 days). (2) A new manual run, "Buy windows", tests whether a time of day or a weekday is reliably cheaper to buy on, using only days it has not learned from.
Nothing else changes: the retail fetcher, the Forever fetcher, the schedule, the forecasts and the app are untouched. Nothing in this zip buys, sells or writes anything except the new hourly files in the private dataset.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py). Overwrite when asked. Do not create a frontend folder.
  engine/daily.py                       replaces the existing file (adds the hourly table; item_daily is unchanged)
  engine/windows.py                     NEW
  tools/lake.py                         replaces the existing file (only mode daily changes: it also writes the hourly day files)
  tools/buy_windows.py                  NEW
  tools/buy_windows.yml                 NEW (copy of the workflow, as with the other workflows)
  .github/workflows/buy_windows.yml     NEW  (this is the one GitHub runs)
  tests/test_windows.py                 NEW
  tests/test_daily.py                   replaces the existing file
  docs/DATA_CONTRACT.md                 replaces the existing file (adds section 2v)
requirements.txt and the other workflows are NOT in this zip and do not change. tools/lake.py was last replaced by v0.7.29; if you have edited it yourself since, tell me before unzipping.

AFTER YOU PUSH (about 15 minutes of clicking, then waiting)
  1. Actions > Daily chain > Run workflow with the defaults. This starts the hourly history (it writes engine/hourly/h_<day>.parquet files in the private dataset). From then on the 06:17 UTC run keeps it growing by itself.
     The newest 12 hours of any run are written by the NEXT run (the flags need 12 hours of context either side), so a day is complete after the following day's run.
  2. Backfill, the same way you backfilled item_daily: Actions > Daily chain > Run workflow with days = 7 and end_date set, one run at a time, each ending where the last began (overlap of a day on purpose):
        2026-11-15, then 2026-11-08, then 2026-11-01, then 2026-10-25     (retail hourly data starts Oct 25, so that is as far back as it goes)
     If a run runs out of time, use days = 4 and more end dates. Running one twice does no harm (days are merged, never shrunk). The forecast-ledger step at the end of each run is also safe to repeat.
  3. Actions > Buy windows > Run workflow with the defaults. Takes a few minutes. The report is in the run's summary and in the file buy_windows.txt (download it from the run page, "Artifacts"). Send me that file.
     Later you can try the other settings: window 2 / 4 hours, or price p15 (what the cheapest 15% of the units cost, closer to what buying a quantity pays than the single cheapest listing).

WHAT TO EXPECT FROM THE FIRST REPORT (so a disappointing one is not a surprise)
  - Hour of day: about 12 later days to grade on (10 days are learning only, 22 days exist). A thin answer that firms up every day.
  - Day of week: needs about 7 weeks of hours, which retail reaches around mid-December. Until then the report says "not enough days yet". The older test on years of daily prices found no weekday rhythm in the typical price; this one tests the LOW price, which is what a buyer pays.
  - "No pattern visible" is a real answer, not a failure. It means the price at the best-looking hour was no cheaper than a random hour on days the test had not seen. Then the money is in buying DIPS against an item's own range, which is the next piece.
  - The "oracle" column is the best hour of each day looked at afterwards. Nobody can know it in advance, and in plain noise it is always positive. It only shows how much there is to find.

WHAT CHANGED, IN WORDS
  - New hourly files: one per US Eastern day, in the private dataset under engine/hourly/. Per item per hour: lowest price, units listed, units listed within 5% and 10% of the lowest price (how many you could buy near it), the "cheapest 15% / 30% of units" prices, and the flags that mark broken or wiped hours. A few MB a day.
  - The test learns the cheapest window only from earlier days, grades it on the next day, and repeats. Savings are shown as a 90% range, never one number. Items are pooled by family (item groups), by price tier and all together, so a pattern has many items to show up in.
  - The resale check asks: if you bought in that window and sold 1, 2 or 3 days later at the median price after the AH cut, how often would 5% be left? (Your Nov 16 rule: the buy price must leave 5%+ profit after the cut.)
