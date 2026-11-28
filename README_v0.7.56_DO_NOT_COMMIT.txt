v0.7.56 (Nov 27): FIX for the 10 regular-auction packs the compaction check refused. The 110 packs already stored are fine and stay.

WHAT HAPPENED
  Your run stored 110 packs (Nov 13-17), each read back byte for byte identical. The red X was the job reporting 10 packs it refused to store, all regular
  auctions. Their hourly files do not always have the same columns: in some hours the "bid" column sits in a different place, and in an hour where no auction
  had a bid there is no "bid" column at all. The pack used the first hour's column list for every hour, so those packs did not rebuild exactly, and the check
  refused them. Nothing wrong was stored and no hourly file was touched.
FIX: each hour's own column list is now kept. Packs already stored (older format) still read back correctly (tested).
The refused days were not marked as packed, so the next run simply packs them again.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker. Overwrite when asked.
  engine/foreverpack.py, tools/forever_pack_probe.py, tests/test_forever_pack.py, docs/DATA_CONTRACT.md

AFTER YOU PUSH
  Run "Forever compact" with save = yes again (days 5 or 60; one at a time). Each run first redoes any refused packs from earlier days, then carries on.
  The exit should be green when every pack passes. Repeat until "days still waiting after this run: 0".
