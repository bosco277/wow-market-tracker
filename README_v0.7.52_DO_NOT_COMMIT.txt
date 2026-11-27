v0.7.52 (Nov 27): AUCTIONATOR EXPORT on the Flips and House flips tabs, and lib.py back in place.

WHAT IT ADDS
  Under the Flips table and under the House flips table: "Export to Auctionator (shopping list)". Open it and it shows one line like
      WMD House flips 116 to 110 Nov 27 1652^"Thick Leather"^"Mooncloth"^"Arcane Crystal"
  (the same format Auctionator exports: the list name, then each item in quotes, joined by ^). Items are in the table's order, best first.
  Copy it (button at the top right of the box), then in game: Auctionator > Shopping > Import, paste, and search the list to check whether the deals still stand.
  The list name carries the tab, the house(s) and the time, so each export is a new list and old ones are easy to tell apart.

ALSO IN THIS ZIP
  lib.py and tests/test_frontend_lib.py, exactly as in v0.7.41 (the files deleted after my wrong note in v0.7.48). The app needs lib.py: it holds the logic of
  every tab. If you already pushed wow-restore-lib.zip, these are the same files; overwriting is harmless.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py and app.py). Overwrite when asked.
  app.py, houseflip.py, lib.py, tests/test_houseflip.py, tests/test_app_smoke.py, tests/test_frontend_lib.py

AFTER YOU PUSH: Streamlit redeploys by itself. Open the Flips or House flips tab and expand "Export to Auctionator" under the table.
