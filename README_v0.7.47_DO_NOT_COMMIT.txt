v0.7.47 (REVISED Nov 26, morning: this zip replaces every earlier v0.7.47 zip; use only this one): Forever DAILY tables, one row per auction house, item and day. WoW Forever only.
  Nov 26 change: only docs/DATA_CONTRACT.md differs from the Nov 25 evening zip (a wrong sentence about compaction was corrected; no code changed).

THE DAILY FILES ARE A SUMMARY, NOT A COMPACTION
  This job only ADDS files named daily/d_YYYY-MM-DD.csv.gz. It never deletes, rewrites or replaces a raw hourly file, so every auction id in the raw files is still there afterwards.
  The daily table is a rebuildable view: it can be thrown away and rebuilt from the raw hourly files at any time (use "rebuild"). It is NOT a substitute for them.
  Packing the raw hourly files (compaction) is a separate step that is NOT built. When it is built it will keep every unique auction id and will be proven lossless before anything is deleted.

WHAT IT DOES
  Every hour we save a raw snapshot of every auction house. This job boils each day down to one small row per house and item, so the app, the buy-below prices and the crafting margins
  can read a day in a moment instead of opening 24 big files per house. Each row says (gold per unit):
    price_day / price_min     the day's typical lowest price, and the lowest price seen
    price_p15 / price_p30     the price at which the cheapest 15% / 30% of the stock is reached (one cheap listing cannot set it)
    buy20 / buy200            the average price per unit to buy 20 / 200 units, walking up the listings
    units_day, listings_day, gold_day     typical units for sale, typical number of listings, gold in play
    hours, flagged_hours      how many good hourly readings the day is based on (18+ = a full day) and how many were left out
    SALES AND SPEED (on by default; see the warning below):
    sold_cert_day             units per day that CERTAINLY sold (a listing still standing with fewer units: only a purchase does that)
    sold_low_day / sold_day / sold_high_day     the conservative, middle and optimistic estimates of units sold per day
    turnover_day              sold_low_day / units_day: the share of the standing stock seen selling per day
    active_share              the share of the day's hours in which any sale was seen: how often the market visibly moves
  WARNING: all the sales columns are LOWER BOUNDS. Blizzard publishes each house once an hour (about :22), so a listing posted and sold inside one hour is in no snapshot, for us or for any other tool.
  If most Forever sales happen within minutes (your experience), most of the volume is invisible. The columns are for comparing items and days and for spotting a market speeding up or slowing down, not for the true number.
  To turn them into true volume we need your own sales and posting times (Journalator / the app ledger) to measure how much of it the hourly snapshots catch. That is the next step on sales, after this runs.
  Commodity items only for now (reagents, herbs, leather, cloth, consumables). Gear and weapons trade as regular auctions: a later step. The report counts how many recipes are already fully priced.
  Stored as one file per day in the Forever dataset: daily/d_YYYY-MM-DD.csv.gz (all houses in it). Only new or changed days are written, in one commit. Empty houses (112, 115, 118) are left out.

ALSO IN THIS ZIP: house 116 (Blackwater, PvP ruleset) is marked "confirmed" in the house map, from your Thick Leather listing. The next "Forever tables" run stores the updated house map.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py and app.py). Overwrite when asked.
  engine/foreverdaily.py                     NEW
  engine/daily.py                            replaces the existing file (one optional argument added; retail results are unchanged, all retail tests pass)
  engine/forevertables.py                    replaces the v0.7.46 file (house 116 confirmed)
  tools/forever_daily.py                     NEW
  tools/forever_daily.yml                    NEW
  .github/workflows/forever_daily.yml        NEW
  tests/test_forever_daily.py                NEW
  tests/test_forever_tables.py               replaces the v0.7.46 file
  docs/DATA_CONTRACT.md                      replaces the existing file (adds the v0.7.47 notes)
requirements.txt and every other file are NOT in this zip and do not change.

AFTER YOU PUSH
  1. Actions > "Forever daily tables" > Run workflow, leave everything as it is (save = no, sales = yes). It only REPORTS. Note how long it takes and send me forever_daily.txt.
     I have NOT seen it run on real data: the speed for about 12 houses over 4 days, the file sizes and whether the spike / blip flags behave on Forever are what the first report tells us.
  2. If the report looks right, run it again with save = yes. That stores the days.
  3. Backfill: run it with save = yes, days = 10 and end_date = 2026-11-21, then again with end_date = 2026-11-11, and so on until the report says it read no files. (Each run leaves out the first and last 12 hours of the files
     it reads, so neighbouring runs fill each other's edges. The days overlap by design; running them twice is safe.)
  4. To make it run by itself every day (13:47 US Eastern in winter), set a repository variable: Settings > Secrets and variables > Actions > Variables > New repository variable.
       Name:  FOREVER_DAILY_BUILD      Value:  yes
     To stop it, delete the variable. Manual runs never depend on it. Do this after step 2 looks good.
  Also still pending from before: the "Forever tables" run on 7.46 with save = yes (that run also stores the updated house map once this zip is pushed).

WHAT TO LOOK AT IN THE REPORT
  Section 1: every house should show 90+ snapshots for 4 days; "gaps>90m" and "skipped" should be 0 or small. A house that FAILED is named.
  Section 2: the newest days are partial (fewer complete rows); days at least two days old should be nearly all complete.
  Section 4: prices of well-known items (Light Leather, Linen Cloth...). Check a few against what you see in game. The 'Visible sales' lines say how much of each market showed any sale at all; if that is very low
             for items you know sell, that is the invisible-sales problem showing up in numbers, and it tells us how big the capture check needs to be.
  Section 5: "both" is how many recipes can already be costed and valued from commodity prices alone.

PRIVACY: nothing in the report names an account or a character (the dataset's name is blanked out of every message, including errors; no secret is ever printed). The job needs only HF_TOKEN, no Blizzard secrets.
