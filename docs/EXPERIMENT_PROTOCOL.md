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

An item's **cycle** is: buy, then list (half "under", half "same", 12 hours), then wait until all of its listings are
gone (sold or expired), fill in the results, and start the next cycle. **Never start a new cycle for an item while it
still has listings up**, because then we cannot tell which listing sold.

**Starting a cycle for an item:**
1. **Buy** the shortlist's `buy_units`. You do not need to write down what you paid. (We can work out the cost from the
   book at that moment.) Buying and posting within a few minutes of each other is exactly right.
2. **List half "under":** **1 silver below the lowest price** (the smallest step the AH allows).
3. **List half "same":** at **exactly the lowest price**.
4. Add one log row per listing (see below). Always set the listing length to **12 hours**.
5. Units that came back unsold from the last cycle: list them too, half and half, and write "relist of returned units"
   in notes.

**Checking in (about twice a day, best a little after your listings expire):** for each item, open the mailbox and My
Auctions and fill in `units_sold` and `units_expired` on that item's rows. Then:
- **Nothing of this item still listed?** Start its next cycle now.
- **Some still listed?** Leave it alone. Do not cancel, buy back or repost it.

Why "under" and "same"? In retail, and per your beta experience in Forever, the **newest** listing at a price is bought
first. So the "same" listing should sell almost as fast as the "under" one. If it does not, something is different from
what we think, and we want to know.

**Odd markets are welcome.** If an item's price has been flipped or has jumped, keep listing it and write "abnormal
market" in notes. I will analyse those cycles separately from the normal ones, and they are exactly the data we need to
learn about flips.

**Mistakes happen** (for example posting the wrong amount at one price and fixing it by buying and posting more). Just
add a row for what you really did and say so in notes.

## What the log can and cannot tell us

You cannot see when a sale happened while you are offline, and mail only shows what sold, not the clock time. That is
fine. **The log tells us how many units really sold from each listing. It does not need to tell us when.** The "when"
comes from the hourly snapshots: your "under" listing sits at its own price, so we can watch it shrink in the data hour by
hour. The log is the answer key we check that against.

## What to write in the log: one row per listing

| Column | What to put |
|---|---|
| date | Day you posted it |
| time_posted | Clock time to the nearest few minutes (it matters more than it sounds; "about 9 PM" is a bit too loose) |
| tz | `EDT` or `EST`. **The clocks go back on Sunday Nov 1** |
| item_id | From the shortlist |
| lane | `UNDER`, `SAME`, or `OTHER` (anything else; say what in notes) |
| list_price | Price per unit in gold, as shown on the AH (for example 1.49) |
| quantity | Units in that listing |
| units_sold | Filled in later: units of this listing that sold (from the sale mails) |
| units_expired | Filled in later: units that came back unsold |
| notes | Anything odd: "abnormal market", "relist of returned units", "price moved before I posted", mistakes |

A partly sold listing is normal: quantity 45, units_sold 30, units_expired 15. Mail gold amounts are not needed (they
appear to include the returned deposit, so I cannot use them).

## Schedule (launch is Wed Nov 4)

Check in about twice a day from now until Tue Nov 3. On Wed Nov 4 collect the last results and **stop buying**.
Items that sell out quickly can run more cycles, which is useful: more cycles means more answers.

It is a small sample, so the numbers it gives us will be rough. They are far better than
guesses, and every number will come with a "how sure are we" attached.

## What I will do with the log

1. Match each cycle's purchase (the units you listed, at the time you posted) to the snapshot interval it fell in, and compare the units the classifier credits with what you really bought.
2. Compare how fast your "under" and "same" listings sold with what each queue rule predicts, which tells us newest-first or oldest-first.
3. Count how many of your listings never appeared in any snapshot (the invisible sales) to put a real number on the blind spot.
4. Adjust the engine's settings (`cancel_share`, the blind-spot correction) from what we learn, re-run the validation, and report the new bounds.

## Stop rules

Stop and tell me if: an item's price jumps by more than 30%, a session loses more than you are comfortable with, or the
data refresh stops (the log output will show gaps). Do not push on.

## Running the reconcile check

Save your log as `experiment_log.csv` in the repo root (same columns as the template; example rows are ignored) and commit it.
Then Actions, "Market tools", mode `reconcile`, days 8. Rows without `units_sold` yet are shown but not counted. Paste me the output.
