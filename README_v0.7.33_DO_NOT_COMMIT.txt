v0.7.33: Forever realm ids can be set in a GitHub setting (no push), plus an alert if the realm list is live but none of our realms are in it.
The retail fetcher and every retail file are untouched.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py). Overwrite when asked. Do not create a frontend folder.
  fetch_forever.py                          replaces the v0.7.32 file, main folder
  .github/workflows/forever_fetch.yml       replaces the v0.7.32 workflow (hidden folder). If your unzip tool skips hidden folders, use tools/forever_fetch.yml and
  tools/forever_fetch.yml                   identical copy: open .github/workflows/forever_fetch.yml in GitHub, edit, paste the contents
  tests/test_forever_fetch.py               replaces the v0.7.32 file
  docs/DATA_CONTRACT.md                     replaces the existing file
requirements.txt is NOT in this zip and does not change.

WHAT CHANGED
  1. Realm ids: the automatic runs now read a repository VARIABLE named FOREVER_REALM_IDS (comma separated numbers). Leave it unset and nothing changes (the built-in list is used).
     A hand-started run can still type ids into the "realm_ids" box; the box wins over the variable.
     To set it: repo > Settings > Secrets and variables > Actions > "Variables" tab > New repository variable. Name: FOREVER_REALM_IDS   Value: e.g. 7001,7003,7005
  2. New alert: if Blizzard's Forever realm list is live but none of the realm ids we want are in it, the run prints the first 25 "id name" entries in the log and sends ONE Discord message
     (when DISCORD_WEBHOOK_URL is set) with the same list and what to do. The run stays green. A live-but-empty list is just logged.
  3. Nothing is collected until real realm ids match. Nothing else about the fetcher changes.
