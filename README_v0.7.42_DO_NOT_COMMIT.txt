v0.7.42: the Forever health check. A manual workflow that reads the Forever dataset and writes one report: what is being collected, whether every hour is there, how fast the data is growing, how it is stored (and whether the engine's own reader takes it), and whether the auction houses are separate markets.

It only reads. Nothing is written to any dataset. The fetcher, its settings, the engine, the app and requirements.txt are not touched.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py and app.py). Overwrite when asked.
  tools/forever_health.py                    NEW
  tools/forever_health.yml                   NEW (copy of the workflow, like the other tools)
  .github/workflows/forever_health.yml       NEW
  tests/test_forever_health.py               NEW
  docs/DATA_CONTRACT.md                      replaces the existing file
requirements.txt and every other workflow and file are NOT in this zip and do not change.

HOW TO RUN IT (after you push)
  1. GitHub > Actions > "Forever health check" > Run workflow. Leave the defaults (compare_hours 48, days 0).
  2. It takes a few minutes: it lists the dataset, reads the newest file of every source, and downloads up to 48 hours of the commodity files (never more than 800 MB).
  3. The report is in the run's summary and in the downloadable file forever_health.txt (Artifacts, "forever-health"). Send me the .txt.
  If you want it quicker, set compare_hours to 24; 0 skips the price comparison between houses.
  It uses the HF_TOKEN secret, the same one the Forever fetcher uses. If it stops with a permission error, that token cannot read the Forever dataset: tell me what the message says.

WHAT TO LOOK AT FIRST (the lines at the top of the report)
  [OK] / [WATCH] / [PROBLEM] lines: how many days are collected and how many hours are missing, how old the newest file is, whether the engine's reader takes the files, whether any houses are copies of each other, and how fast the data grows.
  Section 1 lists every house being collected. This is where we find out whether "all realms and regions" is true: the fetcher's defaults are the US only, one ruleset, every house that has listings; nothing else unless you set repository variables.
  Section 4 has the lift-and-shift test. One catch is already known from the code: the engine calls a snapshot a commodity only when its source is named exactly "commodities", so Forever's "_commodities" sources would be read as regular auctions. The adapter will fix that; the report shows it.
  Section 5 compares the houses: shared auction ids, and the price of the same item in each house. This is the evidence for "one market or separate markets".

PRIVACY: nothing in the report names an account or a character (the dataset's name is blanked out of every message). The report contains market data only. On a public repo, any signed-in GitHub user can download a workflow's artifact, so the same applies to this one.

NOTE: v0.7.41 is already on GitHub; this zip does not include it.
