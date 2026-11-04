v0.7.32: Forever fetcher (off until you switch it on). The retail fetcher and every retail file are untouched.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py). Overwrite when asked. Do not create a frontend folder.
  fetch_forever.py                          NEW, main folder
  .github/workflows/forever_fetch.yml       NEW workflow (hidden folder). If your unzip tool skips hidden folders, use tools/forever_fetch.yml and
  tools/forever_fetch.yml                   NEW, identical copy: upload it in GitHub (Add file > Create new file, name it .github/workflows/forever_fetch.yml, paste the contents)
  tests/test_forever_fetch.py               NEW
  docs/DATA_CONTRACT.md                     replaces the existing file
requirements.txt is NOT in this zip and does not change.

TO TURN IT ON (after pushing, any time before or after launch):
  1. GitHub repo > Settings > Secrets and variables > Actions > "Variables" tab > New repository variable. Name: FOREVER_ENABLED   Value: true
  2. Hugging Face: create a PRIVATE dataset named wow-forever-data-lake under your account (the script also tries to create it itself). Your HF_TOKEN secret must be allowed to write to it
     ("Write" tokens can; a "Fine-grained" token needs the new dataset added).
  3. Test it once by hand: Actions tab > "Forever Auction Extractor" > Run workflow. Before Blizzard serves Forever the log ends with a summary like
     "Forever: Blizzard does not serve the Forever realm list (not live yet)" or "... not served yet 8" and the run is green. That is correct.
  4. After that it runs by itself after every hourly retail extraction. Optional: the DISCORD_WEBHOOK_URL secret you may already have sends ONE message when Forever data first arrives.
TO TURN IT OFF: delete the FOREVER_ENABLED variable (or set it to anything but true).
If you name the dataset differently, tell me (it is the HF_REPO_ID line near the top of fetch_forever.py).
