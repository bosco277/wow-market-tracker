v0.7.48 (Nov 27): the app switched to WoW Forever only, and the buy200 fix.

REVISED Nov 27, 10:30 AM: this zip replaces the first v0.7.48 zip (use only this one). The app is now WoW Forever ONLY: no Game switch, no retail page.

WHAT YOU GET IN THE APP
  The app opens straight on WoW Forever. The retail page is removed from app.py (it is still in your git history if you ever want it back).
  The chosen house is kept in the page address (&house=110), so a bookmark opens the house you want.

  The page reads only the Forever dataset (the daily tables, item names and the house map):
    House picker   "110 · PvP · Alliance" and so on. "(unverified)" marks the neutral houses we have only placed by slot order (117, 121, 125).
    Day picker     US Eastern days. The default is the newest day where most items are complete (the newest stored day is still filling in).
    Market tab     headline numbers, item search, class filter, min gold in play, "complete days only". The table is ranked by gold seen traded per day
                   (seen sold x price): price, lowest, cheapest 30%, buy 20 / 200 average, units, gold in play, seen and certain sales per day, turnover,
                   hours selling. Gold shows 4 decimals because Forever prices are small. Click a row for the item's charts by day, the same item in
                   every house that day, and every stored day as a table.
    Compare houses Alliance, Horde and Neutral houses of the chosen ruleset side by side on the day, biggest price gap first. NO auction house fees are
                   included (the neutral house's cut on Forever is unchecked), so treat small gaps as no gap. Daily prices, not live listings.
    About tab      what every column means.
  Sales numbers are still LOWER BOUNDS (a listing posted and sold inside an hour is in no snapshot); the page says so wherever they appear.

ALSO IN THIS ZIP
  buy200 fix: the deeper buy price is now built from the extra cost of going deeper in each hour, so it can never come out under buy20 (the Cured Rugged
  Hide case), and it is blank unless 200 units were for sale in at least half the day's hours. Same columns, same files.
  Corrected backfill instructions and two tests for them (windows must overlap by 2 days). Nothing to run for that: your backfill is done.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py and app.py). Overwrite when asked. Do not create a frontend folder.
  app.py                                     replaces the existing file: now about 20 lines that open the Forever page (the retail page is removed)
  forever_page.py                            NEW (next to app.py and lib.py)
  forever_lib.py                             NEW (next to app.py and lib.py)
  engine/foreverdaily.py                     replaces the v0.7.47 file (buy200)
  tools/forever_daily.py                     replaces the v0.7.47 file (help text only)
  tools/forever_daily.yml                    replaces (comments only)
  .github/workflows/forever_daily.yml        replaces (comments only)
  tests/test_forever_frontend.py             NEW
  tests/test_forever_daily.py                replaces
  tests/test_app_smoke.py                    replaces (the retail page tests are removed with the page; it checks no retail table is read)
  docs/DATA_CONTRACT.md                      replaces (v0.7.48 notes)
lib.py, requirements.txt and every other file are NOT in this zip and do not change. lib.py (the retail page's helpers) is no longer used by the app;
it is harmless to leave. If you want it gone, delete lib.py and tests/test_frontend_lib.py together in the same commit.

AFTER YOU PUSH
  1. Streamlit redeploys by itself. Open the app: it should open on WoW Forever, house 110, with no switch.
  2. If it says "Could not read the WoW Forever dataset", the app's HF_TOKEN cannot read the Forever dataset. In Hugging Face, give that token read access to
     the Forever dataset (or make a new read token that covers both datasets) and paste it into Streamlit: Manage app > Settings > Secrets, as HF_TOKEN.
     The first load downloads about 15 small day files and takes a few seconds; after that it is cached for 15 minutes.
  3. Tell me what looks wrong or missing. I have NOT seen this page in a real browser or against the real data: the tests run it against stand-ins.
     Things worth checking: a few prices in house 110 against the game, and whether the Compare houses gaps between 110 and 116 look real.
  4. Optional, only if you want the new buy200 on old days now: run "Forever daily tables" with days = 10, save = yes, end_date = 2026-11-25, then again with
     end_date = 2026-11-17 (rebuild stays no). Days that are not re-run keep the old buy200, which is off by under 1% at worst. Nothing uses buy200 yet.

PRIVACY: nothing on the page or in an error names an account; the dataset's name is blanked out of error messages.
