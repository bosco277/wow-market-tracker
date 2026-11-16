v0.7.38: the second half of the buy-window work, after your first report. The Buy windows report now has a DIP TEST (buying when an item falls into the bottom of its own recent range), the time-of-day test can no longer be fooled by a steady price slide, the resale check shows losses next to gains, and the weekday test speaks about three times sooner.
Nothing else changes: the daily run, the hourly files, the retail fetcher, the Forever fetcher, the forecasts and the app are untouched. Nothing here buys or sells anything.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py). Overwrite when asked. Do not create a frontend folder.
  engine/windows.py         replaces the existing file (from v0.7.37)
  engine/dips.py            NEW
  tools/buy_windows.py      replaces the existing file
  tests/test_windows.py     replaces the existing file
  tests/test_dips.py        NEW
  docs/DATA_CONTRACT.md     replaces the existing file (adds the v0.7.38 notes and your Nov 16 first-report findings to section 2v)
The workflows, tools/lake.py, engine/daily.py and requirements.txt are NOT in this zip and do not change. No new backfill is needed.

AFTER YOU PUSH
  1. Actions > Buy windows > Run workflow with the defaults (about a minute). Send me buy_windows.txt as before. It is ONE report now: clock test, weekday test, resale check, then the dip test.
  2. Run it again in 2 days: by then most pools have 10 graded days and the "too few test days" labels start turning into real verdicts. Nothing else to do; the daily run keeps adding days.

HOW TO READ THE NEW DIP SECTION
  - A dip = the price is in the bottom fifth of that item's own last 7 days AND at least 3% under its median of those days. The buy is the first dip hour of each day. It needs no hold-out because the trigger only looks backwards.
  - "Units near floor" = how many units were listed within 5% of the cheapest price at the dip hour. That is the quantity a plan could actually take, so it sizes the buy order.
  - "gain" = bought at the dip and sold 1 / 2 / 3 days later at the median price, with at least 5% left after the cut. "loss" = lost 5% or more. A rule that wins 20% of the time and loses 25% of the time is not an edge.
  - "Edge" = how much cheaper the dip hour was than an average hour of the same days. By itself it says nothing about resale (any low hour is below average). For a CRAFTING INPUT (copper ore now, tin later) it is the saving you capture. For a RESALE, the resale verdict at the end of each row decides: the TYPICAL (median) buy has to clear 5% after the cut and gains have to outnumber losses.
  - The "by how deep the dip was" table shows which dips can pay. A resale needs about a 10.5% rise between buying and selling, so shallow dips (3-6%) cannot, and deep ones can only if the price really comes back. A dip that is a lasting price drop does not come back, and the median margin shows it.

WHAT CHANGED IN THE CLOCK TEST
  - Each hour is compared with the item's median over the 25 hours around it, not the day's median. Your first report compared against the day's median, so a steady fall in prices across days could make evenings look cheap. The new version cannot be fooled that way (tested: a market sliding 4% a day shows a fake 1.5%+ evening saving in the old way and about zero in the new one).
  - The resale check also shows the median margin and the share of buys that lose 5%+. The old average margin was dominated by a few items that spiked and has been dropped from the printout.
  - Item groups: only the best 12 are listed, with a note, because with 100+ groups some look good by luck.
  - Weekday test: grades every test day that lands on the two weekdays it learned as cheapest (before: only days landing on one weekday). It grades nothing before 32 days of history and needs about 53 days (8 weeks) for a verdict: retail around Dec 19, Forever around Jan 5.
