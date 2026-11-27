v0.7.51 (Nov 27): MEASURE BEFORE COMPACTING, and the FIX FOR THE FAILED DERIVE SAVE. (Revised 2:10 PM: replaces the first v0.7.51 zip; use only this one.)

THE SAVE FIX
  Your "Forever derive" save run failed while uploading. Hugging Face's newer storage (Xet) does not accept files handed over from memory, so every
  upload fell back to plain HTTP, and the 97-file upload kept failing part way through. Now:
    - files are written to disk first and uploaded from there, so Hugging Face uses Xet (this also applies to the tables and daily jobs);
    - derive saves each house as soon as it is built, in its own commit, so a failed upload loses that one house, not the whole 30-minute run;
    - an error now shows its real cause, not only "Error while uploading".

WHAT IT ADDS
  1. The Forever derive report now splits each house's time into "load s" (downloading the hourly files) and "calc s" (the derive itself).
     Compaction can only shorten the load part, so this tells us how much faster derive can get.
  2. A new workflow, "Forever pack probe". It takes one day of hourly files for a few houses (every column, every auction id), packs it three ways
     (one day file; one day file sorted by auction id; a listing ledger), and checks for each one that:
       - every hourly file rebuilt from it is EXACTLY the original (every row, every column, every auction id), and
       - the sales calculation on the rebuilt day gives exactly the same sales, item by item and hour by hour.
     Then it prints the sizes and what a year would come to for all 12 houses. Nothing is stored; nothing is deleted.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py and app.py). Overwrite when asked.
  engine/foreverpack.py                     NEW
  tools/forever_pack_probe.py               NEW
  tools/forever_pack_probe.yml              NEW (copy)
  .github/workflows/forever_pack_probe.yml  NEW (the workflow)
  tools/forever_derive.py                   replaces (time split in the report; one commit per house)
  tools/forever_derive.yml                  replaces (comment only)
  .github/workflows/forever_derive.yml      replaces (comment only)
  tools/forever_tables.py                   replaces (uploads from files on disk)
  tests/test_forever_derive.py              replaces
  tests/test_forever_tables.py              replaces
  tests/test_forever_pack.py                NEW
  docs/DATA_CONTRACT.md                     replaces (v0.7.51 notes)

AFTER YOU PUSH
  0. Run "Forever derive" with save = yes once more (or let the Cloudflare Worker start it). The report shows "Saved 12 house(s), one commit each" when it works,
     and the load s / calc s split. Then open the app.
  1. Actions > "Forever pack probe" > Run workflow with the defaults (yesterday, houses 110,116,121). Send me the report and the run time.
  2. Send me the derive report from step 0 too (it has the load / calc split).
  From those two we pick the compaction format, and I build the copy-only compaction.
