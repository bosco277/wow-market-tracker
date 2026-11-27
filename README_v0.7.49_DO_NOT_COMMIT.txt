v0.7.49 (Nov 27): RESTORES THE APP. Undoes the v0.7.48 mistake that replaced your six-tab app with a smaller Forever page.

WHAT THIS DOES
  app.py and tests/test_app_smoke.py go back to exactly the v0.7.41 files you had before v0.7.48: Market, Buy targets, Flips, Forecasts, Experiment and
  Confidence, all working as before (on retail data for now).
  Kept from v0.7.48 (all still good): the buy200 fix in the daily tables, the corrected backfill text, and forever_page.py / forever_lib.py, which stay
  in the repo but are NOT used by the app. Their data helpers (reading the daily tables, house labels, house map) will be reused when the app is moved
  to Forever data.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py and app.py). Overwrite when asked.
  app.py                          the v0.7.41 file, byte for byte
  tests/test_app_smoke.py         the v0.7.41 file, byte for byte
  tests/test_forever_frontend.py  runs the Forever page on its own now that it is not in app.py
  docs/DATA_CONTRACT.md           notes on the mistake, the restore and the plan
Do not delete forever_page.py or forever_lib.py. If you already deleted lib.py or tests/test_frontend_lib.py, put them back: get them from the commit
before v0.7.48 on GitHub (open the file, History, the commit before, "View file", copy the content) or ask me and I will send them.

AFTER YOU PUSH
  Streamlit redeploys by itself. The app should look exactly as it did before v0.7.48.

NEXT (not in this zip): the app moved to Forever data with every tab kept. Plan in DATA_CONTRACT.md, v0.7.49 note.
