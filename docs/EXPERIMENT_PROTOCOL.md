# The buy-and-list experiment (plain-language guide)

## Why we are doing this

The Auction House shows what is for sale, never what sold. Our "how many sell per day" numbers are inferred. You are the
only source of sales we know are real: you know exactly what you bought, what you listed, and what sold.

With your log I can check three things:
1. **Do we notice sales that really happened?** (You buy 60 units. Does the data show about 60 units disappearing?)
2. **Are we right about how fast a listing sells?** (You list at the front and at the back of the queue. Did they sell when we predicted?)
3. **How many sales do the hourly snapshots miss completely?** (Some listings are posted and sold between two snapshots, so they never appear.)

## What it costs

Every buy-then-resell loses about **5% to the AH cut** plus a few percent from buying a bit above the lowest price.
So expect to lose roughly **8% of everything you buy**, not of what you hold. With 6 items at up to 150 gold each, one
session buys about 900 gold, and ten sessions about 9,000 gold bought, so **roughly 700 gold lost over the week**, if
everything sells. Items that do not sell come back to you as stock instead of gold.

Ways to spend less: skip the PM buy (half the cost, half the data), or use 4 items instead of 6.

## Before you start (once)

1. Put the files in your repo (the zip is laid out in the right folders) and commit.
2. GitHub, Actions tab, "Pick experiment items", Run workflow. Tick **plan only** first. It only counts files.
3. Run it again with plan only unticked. It reads the last 3 days from Hugging Face (not Backblaze), checks the data,
   and produces the shortlist (6 items). **Paste me the log output.** I will check it before you spend anything.
4. Make a copy of `experiment_log_template.csv` as a Google Sheet or Excel file.

The log output also says **which minutes of the hour the data refreshes** (for example `:03`). You will do your trading
in the **5 to 10 minutes after** a refresh, so that what you do lands cleanly inside one snapshot interval.

## Each session (AM and PM, about 12 hours apart, 15 minutes)

Set the listing length to **12 hours** every time, so that each batch has finished by your next session.

Do this for each item on the shortlist:

1. **Check last session's listings.** Open your mail / sales history. Write down, for each item, how many units sold,
   how many came back unsold, and the gold you received. (First session: skip this step.)
2. **Buy** the shortlist's `buy_units`. Write down the time, units, and total price you paid.
3. **List half "under".** Post half the units at **1 copper below the lowest price**. Write down the time, units, price.
4. **List half "same".** Post the other half at **exactly the lowest price** (the same price as the cheapest listing).
   Write down the time, units, price.
5. Units that came back unsold in step 1: list them too, half under, half same. Log them as extra LIST rows.

Why these two? We were told the game now sells the **newest** listing at a price first (last in, first out). If that is
true, the "same" listing is just as far forward as the "under" one, and both sell quickly. If it is not true, the "same"
listing sits behind everyone already at that price and barely sells. That difference answers the question for us,
using your own listings.

If the lowest price has only a handful of units on it, or the live price looks different from the cheat sheet,
still do the same two listings, and put what you saw in notes ("lowest price had 40 units").

That is all. **Do not** cancel listings, change prices mid-session, or buy anything else from the shortlist items in between.
The experiment only works if we know exactly what you did and when.

## What to write in the log

One row per action. Columns:

| Column | What to put |
|---|---|
| date | Day you did it |
| time | Clock time you see, to the minute |
| tz | `EDT` or `EST`. **The clocks go back on Sunday Nov 1**, so tick this carefully on Nov 1 and after |
| action | `BUY`, `LIST_UNDER`, `LIST_SAME`, `LIST_OTHER` (anything else; say what in notes), or `RESULT` |
| item_id | From the shortlist |
| units | Units bought, listed, or (for RESULT) sold |
| price_gold | BUY: total gold paid. LIST: price per unit. RESULT: total gold received. Write gold with decimals: 1 silver = 0.01, 1 copper = 0.0001 (so 1g 24s 99c = 1.2499) |
| units_returned | RESULT rows only: units that came back unsold |
| notes | Anything odd: the item vanished from the AH, the price spiked, you were late, you made a mistake |

If you are late or make a mistake, **write it in notes. Do not fix it quietly.** A honest mistake we know about is fine; a hidden one ruins the row.

## Schedule (launch is Nov 4)

| Day | Sessions |
|---|---|
| Fri Oct 30 | Pick items; do the PM session if the shortlist is ready |
| Sat Oct 31, Sun Nov 1, Mon Nov 2, Tue Nov 3 | AM + PM |
| Wed Nov 4 (launch day) | Collect the last results (the Nov 3 PM listings finish by morning). No more buying |

That is up to about **9 sessions**. It is a small sample, so the numbers it gives us will be rough. They are far better than
guesses, and every number will come with a "how sure are we" attached.

## What I will do with the log

1. Match each of your BUY rows to the snapshot interval it fell in, and compare the units the classifier credits with what you really bought.
2. Compare how fast your "under" and "same" listings sold with what each queue rule predicts, which tells us newest-first or oldest-first.
3. Count how many of your listings never appeared in any snapshot (the invisible sales) to put a real number on the blind spot.
4. Adjust the engine's settings (`cancel_share`, the blind-spot correction) from what we learn, re-run the validation, and report the new bounds.

## Stop rules

Stop and tell me if: an item's price jumps by more than 30%, a session loses more than you are comfortable with, or the
data refresh stops (the log output will show gaps). Do not push on.
