v0.7.53 (Nov 27): COPY-ONLY LEDGER COMPACTION for WoW Forever. Nothing is ever deleted.

WHAT IT DOES
  A new workflow, "Forever compact". For every house and both kinds of files (commodity listings and regular auctions), each completed UTC day that is not
  packed yet is packed into ONE ledger file, oldest day first:
      packed/<source>/year=YYYY/month=MM/day=DD/ledger.parquet
  Before a pack is kept, the job reads back the exact bytes it would store and rebuilds every hourly file of that day from them. If even one value in one
  row differs from the original, that pack is NOT stored (the report says why). After storing, every file is read back and must be byte for byte the same.
  The hourly files are never touched. The job has no delete operation in it at all. Nothing reads the packed files yet; that is the next step (derive and the
  daily tables reading packed days plus today's hourly files), and then, later and only by your decision, deleting hourly files that are safely packed.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py and app.py). Overwrite when asked.
  engine/foreverpack.py                  replaces (the stored file format added)
  tools/forever_compact.py               NEW
  tools/forever_compact.yml              NEW (copy)
  .github/workflows/forever_compact.yml  NEW (the workflow)
  tests/test_forever_compact.py          NEW
  docs/DATA_CONTRACT.md                  replaces (notes for v0.7.52 and v0.7.53, and the probe and derive numbers)
  Push v0.7.51 and v0.7.52 first if you have not (this zip includes the newer engine/foreverpack.py anyway).

AFTER YOU PUSH (one step at a time)
  1. Actions > "Forever compact" > Run workflow with save = no, days = 1. It packs and checks the oldest day for all 12 houses, both kinds, and stores
     nothing. This is the first time the real parquet write and read-back run, so this step matters. Send me the report and the run time.
  2. If every pack says lossless "yes": run it with save = yes and the defaults. It works through the backlog (Nov 12 onward), oldest first, and stops
     starting new days after 90 minutes; run it again until the report says "days still waiting after this run: 0".
  3. Then schedule it once a day: either set the repository variable FOREVER_COMPACT_BUILD = yes (runs at 06:43 UTC), or add it to a Cloudflare Worker
     like the derive one with workflow file "forever_compact.yml" and inputs { save: "yes" }.
