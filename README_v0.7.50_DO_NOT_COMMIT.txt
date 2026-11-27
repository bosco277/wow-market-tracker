v0.7.50 (Nov 27): YOUR APP ON WOW FOREVER DATA, every tab kept (Phase 1). Push this on top of v0.7.49.

WHAT CHANGES IN THE APP
  Same app, same tabs, now reading WoW Forever data for one auction house at a time.
  At the top: Ruleset (PvP, PvE, RP, RP-PvP), then Auction house ("110 · Alliance", "113 · Horde", "116 · Blackwater (neutral)").
    Every house with data is there. "(unverified)" only on houses not yet confirmed by a listing test (117, 121, 125). 116 shows as confirmed.
    The choice is kept in the page address (?ruleset=pvp&house=110), so a bookmark opens the same house.
  Tabs: Market, Buy targets, Flips, House flips (NEW), Forecasts, Confidence. Experiment is removed.
    Market, Flips, House flips and Confidence work on Forever as soon as the derive job has run (below).
    Buy targets and Forecasts are kept and say "not built for WoW Forever yet": they come in the next phases.
  Gold now shows 4 decimals (down to the copper), because Forever prices are small.

HOUSE FLIPS (new tab)
  Pick your Alliance or Horde house. The tab pairs it with the Blackwater (neutral) house of the same ruleset.
  Direction: "Buy on Blackwater, resell on Alliance/Horde" (default) or the reverse.
  Each row: what is listed on the buy house now at or under the most you can pay, how many units to buy (capped at what the resell house sells in the
  chosen days), cost, resell price, profit and margin AFTER the auction house cut, sales per day and days to sell there.
  Defaults (adjustable on the tab): 15% margin. Reverse: 30%, or 15% for items seen selling in 75%+ of Blackwater's hours.
  The assumed cuts (5% Alliance/Horde, 15% Blackwater; Classic values, NOT checked on Forever) are subtracted automatically and listed in a note at
  the bottom of the page, not in the table. Worth checking the real cut in game once: it decides which gaps are real.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py and app.py). Overwrite when asked.
  app.py                                   replaces (Forever data, picker, House flips tab, footnote; the tab code is the same)
  houseflip.py                             NEW (next to app.py)
  forever_lib.py                           replaces (picker helpers added)
  tools/forever_derive.py                  NEW
  tools/forever_derive.yml                 NEW (copy)
  .github/workflows/forever_derive.yml     NEW (the workflow)
  tests/test_forever_derive.py             NEW
  tests/test_houseflip.py                  NEW
  tests/test_app_smoke.py                  replaces
  docs/DATA_CONTRACT.md                    replaces (v0.7.50 notes)
  Nothing is deleted. lib.py stays (the app still uses it).

AFTER YOU PUSH (one step at a time)
  1. Actions > "Forever derive" > Run workflow with the defaults (hours 96, houses all, save no). It only reports. Send me out/forever_derive.txt
     (or the run summary) and how long it took.
  2. If it looks right, run it again with save = yes. That stores the tables the app reads.
  3. Open the app. It should open on PvP / 110 · Alliance with the Market tab filled in. If it says "No tables for house 110 yet" after step 2, the app's
     HF_TOKEN cannot read the Forever dataset: give that token read access (Streamlit: Manage app > Settings > Secrets, HF_TOKEN).
  4. Then decide how often it should refresh: set the repository variable FOREVER_DERIVE_BUILD = yes for every 6 hours. Each run uses Actions minutes
     (step 1 tells us how many); we can make it more frequent if the minutes allow.

NOT CHECKED YET: real data. The tests run the job and the app on pretend houses, including the app reading the job's real output. How long a 12-house run
takes, and how many items pass the Forever cut-offs (an item is tracked if it could have sold 2+ a day), will show in step 1.
PRIVACY: nothing in the report or the app names an account; the dataset's name is blanked out of messages.
