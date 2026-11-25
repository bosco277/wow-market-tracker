v0.7.43: the Forever adapter. It reads the stored Forever files into the engine one auction house at a time, and fixes the three things the health check found:
  1. Commodity sources (names ending _commodities) were being read as regular auctions. The adapter sets the kind from the name.
  2. One repeated auction id (seen once, house 114) made the engine reject a whole snapshot. The adapter drops repeated ids and counts them, and counts the ones whose copies disagreed.
  3. Every snapshot now carries its house, and the load stops if a file says it belongs to another house.
It also builds the same hourly table the retail history uses, with a house column in front (only when asked, and it is never uploaded in this version).

Nothing is written to any dataset. The fetcher, the retail loader and reader (loader.py, schema.py), the app, every other workflow and requirements.txt are not touched.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py and app.py). Overwrite when asked.
  engine/forever.py                          NEW
  tools/forever_adapter.py                   NEW
  tools/forever_adapter.yml                  NEW (copy of the workflow, like the other tools)
  .github/workflows/forever_adapter.yml      NEW
  tests/test_forever_adapter.py              NEW
  docs/DATA_CONTRACT.md                      replaces the existing file (adds the v0.7.43 notes and the saved cross-market sniper idea)
requirements.txt and every other workflow and file are NOT in this zip and do not change.

HOW TO RUN IT (after you push)
  1. GitHub > Actions > "Forever adapter check" > Run workflow. Defaults: houses 110,113,116 (the PvP set), days 2, tables no.
  2. First run: leave the defaults. It counts the files first and stops by itself if the request is over 1,500 files or 800 MB.
  3. Second run: set tables to yes. That builds the house-keyed hourly table and describes it (rows, items, days, flag shares per house). It takes longer because it runs the sales classifier.
  4. The report is in the run's summary and in the downloadable file forever_adapter.txt (Artifacts, "forever-adapter"). Send me the .txt.
  It uses the HF_TOKEN secret, the same one the health check uses.

WHAT TO LOOK AT FIRST
  Section 1: files and snapshots per house and kind. Houses 112, 115 and 118 never return data, so they show as "no files" if you ask for them (houses = all).
  Section 2: the kind check. The commodity snapshot count should equal the number of commodity files read.
  Section 3: repeated auction ids. This is the first real count of how often they happen. "Every repeated id was an exact copy" means the choice of copy does not matter; if some copies disagreed, tell me how many.
  Section 4: gaps of more than 90 minutes. A Tuesday gap at the same time as before (14:32 to 16:32 UTC) would fit the weekly maintenance idea.

PRIVACY: nothing in the report names an account or a character (the dataset's name is blanked out of every message, including error messages). The report contains market data only. On a public repo, any signed-in GitHub user can download a workflow's artifact, so the same applies here.

ALSO IN THIS ZIP: docs/DATA_CONTRACT.md now has a "Planned, saved for later" entry for the cross-market sniper (Alliance or Horde toggle, faction to neutral and neutral to faction). Nothing is built for it yet.

NOTE: v0.7.42 should already be pushed before this one (the data contract in this zip includes its notes).
