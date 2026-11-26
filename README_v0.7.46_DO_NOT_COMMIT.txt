v0.7.46: the daily refresh of the Forever tables, plus the fixes found in the first run. WoW Forever only.

It REPLACES the five v0.7.45 files of the same names (same placement). If you have not pushed v0.7.45 yet, push this one instead: it contains everything in 7.45.

WHAT CHANGED
  1. Daily refresh, cheap. Normal days read only the recipe lists (about 27 Blizzard calls) and look up only NEW recipes and NEW item ids. A day with no news costs about 30 calls and a minute or two of Actions time.
  2. Patch watch. Blizzard's Forever data carries a version tag (now 1.60.1, build 70399). The tool remembers it. When it changes (a Forever patch) every recipe and item is read again, and the report says so.
  3. Forever wins over Classic Era. Forever's own item data is incomplete today, so about 3,000 items are named from Classic Era. Once a week the tool asks Forever again about just those, and any that Forever now names switch to Forever's version.
  4. The placeholder "Test Profession [DNT]" is left out. Recipes now keep their "rank".
  5. The report ends with how many Blizzard calls the run made.
  A small file, tables/forever_tables_meta.json, remembers the version and the last re-check. It is written with the tables.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py and app.py). Overwrite when asked.
  engine/forevertables.py                    replaces the v0.7.45 file
  tools/forever_tables.py                    replaces the v0.7.45 file
  tools/forever_tables.yml                   replaces the v0.7.45 file
  .github/workflows/forever_tables.yml       replaces the v0.7.45 file
  tests/test_forever_tables.py               replaces the v0.7.45 file
  docs/DATA_CONTRACT.md                      replaces the existing file (adds the v0.7.46 notes)
requirements.txt and every other workflow and file are NOT in this zip and do not change.

AFTER YOU PUSH
  1. If the tables are not stored yet: Actions > "Forever tables" > Run workflow > save = yes (as in my last message). If they are stored from 7.45: run it once with save = yes. It reads the recipes in full once (a few minutes),
     does the first Forever re-check of the Classic-Era-named rows, and writes the small meta file. Send me forever_tables.txt.
  2. Then, to turn the DAILY run on: GitHub > Settings > Secrets and variables > Actions > Variables tab > New repository variable.
       Name:  FOREVER_TABLES_DAILY      Value:  yes
     The workflow then runs by itself every day at 09:47 UTC (4:47 or 5:47 AM Eastern) and saves. To stop it, delete the variable (or change the value). Manual runs never depend on this variable.
  If a daily run ever fails, GitHub emails you as it does for your other workflows; the stored tables are not changed by a failed run.

WHAT TO LOOK AT IN A DAILY REPORT
  "Static data version: ... CHANGED" means a Forever patch: everything was read again; that run takes longer.
  "(0 read from Blizzard, N reused ...)" is a quiet day. "Forever re-check ... N are now named by Forever" is the weekly upgrade.
  The last line, "Blizzard calls this run", should be small on quiet days (tens, not thousands).

PRIVACY: nothing in the report names an account or a character (the dataset's name is blanked out of every message, including errors; no secret is ever printed).
