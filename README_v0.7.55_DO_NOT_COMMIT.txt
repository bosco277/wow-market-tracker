v0.7.55 (Nov 27): HOURLY LATEST LISTINGS for WoW Forever (current prices and House flips order book, refreshed every hour).

WHAT IT DOES
  A new workflow, "Forever latest listings". For each house it reads only the newest hourly snapshot and stores two small tables per house:
    now.parquet       lowest price now, units and listings now, and whether that price is cheap / normal / expensive for the item (vs its last 72 hours)
    book_now.parquet  the order book (what House flips buys from)
  The app lays these over the daily build's numbers whenever they are newer, so:
    - Market: "Lowest price now", units listed and "cheap / expensive for this item" are at most about an hour old
    - House flips: the listings you buy from and the resell price are at most about an hour old
  Sales per day, typical ranges, confidence and the charts still come from the daily "Forever derive" build. A caption under the title says the time of each.
  It never overwrites the daily build's tables, so the two jobs cannot undo each other. Nothing is deleted.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py and app.py). Overwrite when asked.
  app.py, forever_lib.py                       replace (the overlay and the caption)
  tools/forever_latest.py                      NEW
  tools/forever_latest.yml                     NEW (copy)
  .github/workflows/forever_latest.yml         NEW (the workflow)
  tests/test_forever_latest.py                 NEW
  tests/test_app_smoke.py                      replaces
  docs/DATA_CONTRACT.md                        replaces

AFTER YOU PUSH (one step at a time; it can run while compaction is running)
  1. Actions > "Forever latest listings" > Run workflow with save = no. Send me the report and the run time.
  2. If it looks right, run it with save = yes and open the app: a second caption under the title says "from the hourly refresh, as of ...".
  3. Schedule it every hour: either the repository variable FOREVER_LATEST_BUILD = yes (runs at :45 each hour), or a new Cloudflare Worker like the derive one with
     workflowFile "forever_latest.yml", body inputs { houses: "all", save: "yes" }, and the cron 45 * * * * under Triggers.
     Each run uses Actions minutes, 24 runs a day: the run time from step 1 tells us the monthly total.
