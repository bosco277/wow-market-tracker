v0.7.35: "Forever Find Listing": finds which Forever auction house holds the listing you posted in game (1 Linen Cloth, 5g 43s 21c = 54321 copper).
Nothing existing is changed: fetch.py, fetch_forever.py and every workflow stay as they are. These are NEW files plus the docs.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py). Overwrite docs/DATA_CONTRACT.md when asked. Do not create a frontend folder.
  forever_find.py                           NEW, main folder
  .github/workflows/forever_find.yml        NEW workflow (hidden folder). If your unzip tool skips hidden folders, use tools/forever_find.yml:
  tools/forever_find.yml                    identical copy: in GitHub choose Add file > Create new file, name it .github/workflows/forever_find.yml, paste the contents
  tests/test_forever_find.py                NEW
  docs/DATA_CONTRACT.md                     replaces the existing file
requirements.txt is NOT in this zip and does not change. No new secrets: it uses only BLIZZARD_CLIENT_ID and BLIZZARD_CLIENT_SECRET.

HOW TO USE (after pushing)
  1. Actions tab > "Forever Find Listing" (left list) > Run workflow. Leave the boxes as they are (item 2589, price 54321, any quantity) and start it.
     It reads Blizzard's live data, so it does NOT wait for our hourly save, only for Blizzard to refresh (not confirmed for Forever; allow up to an hour after you posted at about 8:27 AM ET Friday).
  2. Open the run > "Find the listing" step. Send me the whole output.
     - "FOUND" + "Result: the listing is in house N" = that house is your market (your ruleset and faction).
     - "NOT FOUND in any house yet" = Blizzard has not refreshed since you posted (the "data as of" times show when each house was last updated; compare with your posting time), so run it again later.
       If it still does not show after a couple of refreshes, the "closest:" lines show what Linen Cloth is listed at, which tells us whether the price or item was entered differently.
  3. The run costs about 24 downloads of auction data. Run it when you expect the listing to be visible, not every few minutes.
  4. Do NOT cancel or buy your listing until we have the answer (it is only visible while it is up).
