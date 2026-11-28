v0.7.54 (Nov 27): FIX for the one pack the compaction check refused (house 125, regular auctions, Nov 13). Nothing had been stored.

WHAT WENT WRONG
  Some regular auctions are bid-only (no buyout price). In an hour whose file has one, the buyout column reads as decimals (a missing value is a decimal
  "not a number"); in an hour without one, it reads as whole numbers. The pack restored every hour with the FIRST hour's column types, so one hour came back
  as decimals where the original had whole numbers. The values were identical, but the check demands an exact match, so it refused the pack, as it should.
FIX: the column types are now kept for each hourly file. A test reproduces the exact case.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker. Overwrite when asked.
  engine/foreverpack.py, tools/forever_pack_probe.py, tests/test_forever_pack.py, docs/DATA_CONTRACT.md

AFTER YOU PUSH
  1. Run "Forever compact" again with save = no, days = 1. All 24 packs should say lossless "yes".
  2. Then save = yes with the defaults. One day takes about 10 minutes for all houses and both kinds, so the 14-day backlog needs two runs
     (each stops starting new days after 90 minutes). Run it until "days still waiting after this run: 0".
  3. Then schedule it once a day (FOREVER_COMPACT_BUILD = yes, or a Cloudflare Worker with workflow "forever_compact.yml" and inputs { save: "yes" }).
