v0.7.45: WoW Forever recipe tables, item names and the house map. Forever only: nothing here reads retail data.

What it builds (all from Blizzard's Forever data, stored in your Forever dataset as small files under tables/):
  1. RECIPES   every profession -> its recipes -> what each recipe uses (reagents) and what it makes. Forever's own data is the only source: Classic Era does not serve professions.
  2. ITEMS     a name and details for every item on the auction houses and every item a recipe uses or makes. Forever's own data first, Classic Era for the ids Forever does not name.
               Each row says which source named it. Retail is not used. Later runs only look up NEW ids, so they are quick.
  3. HOUSE MAP which auction house is which faction and ruleset, with a "how sure" label on every row (the neutral houses are marked "assumed", because they are not verified yet).

SAFE BY DEFAULT: the workflow has a "save" box that starts at "no". With "no" it only reports what it would store. Nothing is written until you run it again with save = yes.

PLACEMENT: unzip into the MAIN folder of wow-market-tracker (next to fetch.py and app.py). Overwrite when asked.
  engine/forevertables.py                    NEW
  tools/forever_tables.py                    NEW
  tools/forever_tables.yml                   NEW (copy of the workflow, like the other tools)
  .github/workflows/forever_tables.yml       NEW
  tests/test_forever_tables.py               NEW
  docs/DATA_CONTRACT.md                      replaces the existing file (adds the v0.7.45 notes; corrects a timing remark in the v0.7.44 notes)
requirements.txt and every other workflow and file are NOT in this zip and do not change.

HOW TO RUN IT (after you push)
  1. GitHub > Actions > "Forever tables" > Run workflow. Leave everything at the defaults (parts all, save no, retry_days 3).
  2. It takes a while the first time (roughly 17,000 Blizzard lookups, paced at 20 a second: allow 15 to 25 minutes). Blizzard allows 36,000 an hour, so do not start another big Blizzard run in the same hour.
  3. Send me forever_tables.txt (Artifacts, "forever-tables"). I will check that recipes, reagents and names look right BEFORE anything is stored.
  4. Then run it again with save = yes. That writes four files to your Forever dataset in one commit:
       tables/forever_recipes.csv.gz   tables/forever_reagents.csv.gz   tables/forever_items.csv.gz   tables/forever_house_map.json
     Running it later with save = yes is cheap, and a table that did not change is not written again.
  It uses secrets you already have: HF_TOKEN (the same one the Forever fetcher uses, which can write), BLIZZARD_CLIENT_ID, BLIZZARD_CLIENT_SECRET.

WHAT TO LOOK AT FIRST
  Section 1: the table of professions: recipes, how many list reagents, how many name the item they make. The "Sample recipe record" lines show Blizzard's raw answer; if reagent counts look wrong, that is where I
             will see why. "Recipes a crafting tool can price today" is the number that matters for crafting.
  Section 2: "Now N of M ids are named". Last time Forever plus Classic Era named 96.4% of the auction-house ids. The unnamed ones in recipes are called out.
  Section 3: the house map in one line.
  The last section says what is new or changed and whether it was saved.

PRIVACY: nothing in the report names an account or a character (the dataset's name is blanked out of every message, including errors; no secret is ever printed).

NOTE: v0.7.44 should already be pushed before this one.
