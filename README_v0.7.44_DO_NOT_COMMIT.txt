v0.7.44: the Forever probe. One manual workflow that answers three open questions in a single read-only run:
  1. STRUCTURE    What does Blizzard's Forever API say about how markets are laid out? (rulesets, auction houses, whether a house's record states its own ruleset, whether any realm route exists)
  2. ITEMS        Which item ids in the Forever auction houses have names and details in Blizzard's static data (Forever's own first, then Classic Era, Classic, retail), how many have none, and what the
                  gaps are worth in gold. Also: does any source serve professions and recipes (reagent lists, needed for crafting).
  3. FINGERPRINT  Which of our auction houses is which? foreverwowcraft.com lists eight US pages (PvE / PvP / RP / RP PvP, Alliance and Horde) with prices. Those purchases are priced from each of
                  our houses; the house whose prices line up is that page's house. A method check comes first: the PvP Alliance page must come out as house 110, the one house you confirmed in game.

Nothing is written to any dataset. The fetcher, the retail loader and reader, the app, every other workflow and requirements.txt are not touched.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py and app.py). Overwrite when asked.
  engine/foreveritems.py                     NEW
  engine/housemap.py                         NEW
  tools/forever_probe.py                     NEW
  tools/forever_probe.yml                    NEW (copy of the workflow, like the other tools)
  tools/fwc_fingerprint.json                 NEW (the site prices the fingerprint compares against; a capture from Nov 25)
  .github/workflows/forever_probe.yml        NEW
  tests/test_forever_probe.py                NEW
  docs/DATA_CONTRACT.md                      replaces the existing file (adds the v0.7.44 notes)
requirements.txt and every other workflow and file are NOT in this zip and do not change.

HOW TO RUN IT (after you push). Run it SOON: the site prices are from Nov 25 and the fingerprint only works while our books still look like they did then.
  1. GitHub > Actions > "Forever probe" > Run workflow. Leave the defaults (sections all, hours 40, all_rulesets no).
  2. It uses three secrets that already exist: HF_TOKEN, BLIZZARD_CLIENT_ID, BLIZZARD_CLIENT_SECRET.
  3. Expect it to take a while (the structure part downloads every house's full auction list from Blizzard). If you want the quick answers first, run it twice: sections = items,fingerprint first (a few minutes),
     then sections = structure on its own.
  4. The report is in the run's summary and in the downloadable file forever_probe.txt (Artifacts, "forever-probe"). Send me the .txt.

WHAT TO LOOK AT FIRST
  Section 3, the first [OK] / [PROBLEM] line: the method check. If it says PROBLEM, ignore the other lines of that section (the method does not work on this data) and tell me.
  Section 3, the [OK] lines: each is a page matched to a house, and how far ahead of the next house it is. [WATCH] / UNCLEAR means the prices do not separate the houses (for example two houses with the same prices),
  or the match is poor (the site's numbers are approximate). The neutral houses are not on the site; they are in the table as controls, so a neutral house winning a page means the match is wrong.
  Section 2: "named" is the share of item ids Blizzard gives a name to. The list of unnamed ids, biggest gold first, is what a gap-fill (an in-game addon dump) would have to cover.
  Section 1: the lines "FOLLOWS THE URL" / "same whichever URL" answer whether any house record states its own ruleset.

PRIVACY: nothing in the report names an account or a character (the dataset's name is blanked out of every message, including errors; no secret is ever printed). The report contains market data only.

NOTE: v0.7.43 should already be pushed before this one (the data contract in this zip includes its notes).
