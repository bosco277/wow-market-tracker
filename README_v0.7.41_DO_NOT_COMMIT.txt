v0.7.41: fixes the "Tavern Mug" problem on the Buy targets tab (a target 99% under an item's median, with an inflated chance and a Worth that dominated the total).

What it does not touch: the retail fetcher, the Forever fetcher, the hourly files, the forecasts, the other tabs, the daily chain workflow, and requirements.txt.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py and app.py). Overwrite when asked. Do NOT create a frontend folder.
  app.py                          replaces the existing file (v0.7.40)
  lib.py                          replaces the existing file (v0.7.40)
  engine/targets.py               replaces the existing file (v0.7.40)
  tests/test_targets.py           replaces the existing file
  tests/test_frontend_lib.py      replaces the existing file
  tests/test_app_smoke.py         replaces the existing file
  docs/DATA_CONTRACT.md           replaces the existing file
requirements.txt and every workflow are NOT in this zip and do not change.

AFTER YOU PUSH
  1. The app part works at once: extreme discounts disappear from the list as soon as the app redeploys (use the menu, top right > Clear cache, or wait 10 minutes).
  2. The better "chance" numbers need the new tables: Actions > Daily chain > Run workflow, defaults. It is safe to run twice. Until it runs, the chances are the old ones.
  3. After that, the daily schedule keeps everything current by itself.

WHAT CHANGED ON THE TAB
  - Targets more than 50% under the item's 7-day median are left out by default. Under the filters a line says how many were left out ("12 more items would pass these filters but have a target more than 50% under their median"). Tick "Include extreme discounts" to see them.
  - Every target 35% or more under the median has a note in a new "Note" column telling you to check in game first. 35-50% = "Very deep"; 50% and over = "Extreme". The note is a prompt to verify, not a claim that the price is wrong.
  - The "Depth" bands, in the reliability table and in the chances, are now: under 3%, 3-10%, 10-20%, 20-35%, 35-50%, 50%+. Before, everything over 20% shared one band, so a 99%-under target was given the hit rate of a 21%-under one. Each band now has its own. A band with fewer than 30 graded item-days shows no chance rather than a shaky one.
  - The headline "Gold you would save" is now "Combined worth of the list", with a help text that says it assumes you bought every item's whole usual pile: a size gauge, not a total you could actually save.
  - Not changed: the targets themselves, and the Worth formula.

EXPECT after the Daily chain run: chances for the deepest targets drop sharply (a one-off dip is rarely met again), and some items move from a high chance to "no chance yet". That is the fix working, not a bug.
