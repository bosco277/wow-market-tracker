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

Timing note: you do **not** need to hit any particular minute. Write down the exact time of everything you do and I
will line it up with the snapshots afterwards.

## How it works: one cycle per item

An item's **cycle** is: BUY, then LIST (half "under", half "same", 12 hours), then wait until all of its listings are
gone (sold or expired), collect, and start the next cycle. **Never start a new cycle for an item while it still has
listings up**, because then we cannot tell which listing sold.

Check in about twice a day. Best is a little **after your listings expire** (listings posted at 9pm run out at 9am).
Each time you check in, do this for each item on the shortlist:

1. **Look at the mailbox and My Auctions, and log a RESULT row** (details below).
2. **Nothing of this item still listed?** Start its next cycle now:
   - **Buy** the shortlist's `buy_units`. Write down the time, units and total gold paid.
   - **List half "under":** at **1 copper below the lowest price**. Write down time, units, price.
   - **List half "same":** at **exactly the lowest price**. Write down time, units, price.
   - Units that came back unsold: list them too, half and half, as extra LIST rows.
3. **Still some of it listed?** Leave it alone. Do not cancel it, buy it back, or repost it. It will expire by itself
   and you collect it at your next check-in.

Always set the listing length to **12 hours**.

Why "under" and "same"? In retail, and per your beta experience in Forever, the **newest** listing at a price is bought
first. So the "same" listing should sell almost as fast as the "under" one. If it does not, something is different from
what we think, and we want to know.

If the lowest price has only a handful of units on it, or the live price looks different from the cheat sheet, still do
the same two listings, and put what you saw in notes ("lowest price had 40 units").

**Do not** change prices mid-cycle or buy more of the shortlist items in between. The experiment only works if we know
exactly what you did and when.

## What the log can and cannot tell us

You cannot see when a sale happened while you are offline, and mail only shows what sold, not the clock time. That is
fine. **The log tells us how many units really sold in each cycle. It does not need to tell us when.** The "when" comes
from the hourly snapshots: your "under" listing sits at its own price (1 copper below everyone), so we can watch it shrink
in the data hour by hour. The log is the answer key we check that against.

## What to write in the log

One row per action. Columns:

| Column | What to put |
|---|---|
| date | Day you did it |
| time | Clock time you see, to the minute |
| tz | `EDT` or `EST`. **The clocks go back on Sunday Nov 1**, so tick this carefully on Nov 1 and after |
| action | `BUY`, `LIST_UNDER`, `LIST_SAME`, `LIST_OTHER` (anything else; say what in notes), or `RESULT` |
| item_id | From the shortlist |
| units | BUY / LIST: units bought or listed. RESULT: units that **sold since your last check**, from the sale mails (add up the quantities) |
| price_gold | BUY: total gold paid. LIST: price per unit. RESULT: total gold received since your last check. Write gold with decimals: 1 silver = 0.01, 1 copper = 0.0001 (so 1g 24s 99c = 1.2499) |
| units_returned | RESULT only: units that came back unsold by mail (expired) since your last check |
| units_still_listed | RESULT only: units of this item still on the AH in My Auctions right now |
| notes | Anything odd. If the sale mails show different prices for one item, say so (for example "two sales: 20 at 0.9999 and 5 at 1.0000"). The order the mails arrived in is a nice extra if it is easy, not required |

Partial sales are expected. A listing of 45 that sold 30 is just "30 sold" plus 15 either still listed or returned.

If you are late or make a mistake, **write it in notes. Do not fix it quietly.** A honest mistake we know about is fine; a hidden one ruins the row.

## Schedule (launch is Wed Nov 4)

Check in about twice a day from now until Tue Nov 3. On Wed Nov 4 collect the last results and **stop buying**.
Items that sell out quickly can run more cycles, which is useful: more cycles means more answers.

It is a small sample, so the numbers it gives us will be rough. They are far better than
guesses, and every number will come with a "how sure are we" attached.

## What I will do with the log

1. Match each of your BUY rows to the snapshot interval it fell in, and compare the units the classifier credits with what you really bought.
2. Compare how fast your "under" and "same" listings sold with what each queue rule predicts, which tells us newest-first or oldest-first.
3. Count how many of your listings never appeared in any snapshot (the invisible sales) to put a real number on the blind spot.
4. Adjust the engine's settings (`cancel_share`, the blind-spot correction) from what we learn, re-run the validation, and report the new bounds.

## Stop rules

Stop and tell me if: an item's price jumps by more than 30%, a session loses more than you are comfortable with, or the
data refresh stops (the log output will show gaps). Do not push on.
