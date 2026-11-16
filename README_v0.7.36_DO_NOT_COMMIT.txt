v0.7.36: the Forever fetcher rides out short Hugging Face outages (a Hugging Face "500 Internal Server Error" on Nov 16 failed one hourly run), plus the data-contract notes from the Nov 13 market test.
Nothing else changes: same files, same schedule, same settings. The retail fetcher is untouched.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py). Overwrite when asked. Do not create a frontend folder.
  fetch_forever.py                          replaces the existing file, main folder
  tests/test_forever_fetch.py               replaces the existing file
  docs/DATA_CONTRACT.md                     replaces the existing file (adds: the Nov 13 finder result, Forever listing durations 2 / 8 / 24 hours, this retry change)
requirements.txt and the workflows are NOT in this zip and do not change.

WHAT CHANGED
  - Saving to Hugging Face is now tried 5 times over about 4 minutes (waits of 15, 30, 60, 120 seconds) instead of 3 times over 6 seconds. Reading the markers file is tried 4 times (5, 10, 20 seconds).
  - Only temporary trouble is retried (Hugging Face 5xx, "slow down" 429, network errors). A wrong token or missing permission still fails at once.
  - If Hugging Face is still down after all that, nothing changes from before: the run ends red, nothing is saved, one Discord alert goes out, and the next hourly run fetches the same snapshot again (Blizzard refreshes about :22 each hour, so a retry within 30 minutes sees the same data).
