"""The WoW Forever view of the app (app.py calls render() when the Game switch says WoW Forever). Reads only the Forever dataset; retail tables are not touched.

Data: daily/d_YYYY-MM-DD.csv.gz (one row per house, item and US Eastern day), tables/forever_items.csv.gz (names), tables/forever_house_map.json.
Secrets: HF_TOKEN (needs read access to the Forever dataset); optional FOREVER_HF_REPO. For a local run, FOREVER_LOCAL_DIR points at a folder with the same layout.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import forever_lib as F


def _read_file(path):
    with open(path, "rb") as f:
        return f.read()

MAX_DAYS = 60                   # the newest 60 stored days are loaded (about 0.4 MB each)
INK = "#2B4C59"


def _secret(name):
    try:
        v = st.secrets.get(name)
    except Exception:
        v = None
    return v or os.environ.get(name)


def repo_name() -> str:
    return _secret("FOREVER_HF_REPO") or F.DEFAULT_REPO


@st.cache_data(ttl=900, show_spinner="Loading the WoW Forever tables...")
def load_forever(repo: str, max_days: int = MAX_DAYS) -> dict:
    """{daily, items, hmap, error}. Never raises: a problem comes back as a plain message with the dataset's name blanked out."""
    out = dict(daily=F.read_daily([]), items=F.read_items(None), hmap=F.parse_house_map(None), error=None)
    local = os.environ.get("FOREVER_LOCAL_DIR")
    try:
        if local:
            folder = os.path.join(local, F.DAILY_DIR)
            keys = sorted(k for k in (os.listdir(folder) if os.path.isdir(folder) else []) if F.day_of_key(k))[-max_days:]
            blobs = [_read_file(os.path.join(folder, k)) for k in keys]
            def read(key):
                p = os.path.join(local, key)
                return _read_file(p) if os.path.exists(p) else None
        else:
            from huggingface_hub import HfApi, hf_hub_download
            token = _secret("HF_TOKEN")
            api = HfApi(token=token)
            try:
                entries = list(api.list_repo_tree(repo, path_in_repo=F.DAILY_DIR, repo_type="dataset"))
            except Exception as e:
                if "EntryNotFound" not in type(e).__name__:      # no daily/ folder yet is "nothing stored", not a token problem
                    raise
                entries = []
            keys = sorted(e.path for e in entries if F.day_of_key(getattr(e, "path", "")))[-max_days:]

            def fetch(key):
                return _read_file(hf_hub_download(repo, key, repo_type="dataset", token=token))
            with ThreadPoolExecutor(max_workers=8) as ex:
                blobs = list(ex.map(fetch, keys))

            def read(key):
                try:
                    return fetch(key)
                except Exception:
                    return None
        out["daily"] = F.read_daily(blobs)
        out["items"] = F.read_items(read(F.ITEMS_KEY))
        out["hmap"] = F.parse_house_map(read(F.HOUSE_MAP_KEY))
    except Exception as e:
        out["error"] = F.scrub(f"{type(e).__name__}: {e}", repo)[:400]
    return out


def column_config(df: pd.DataFrame) -> dict:
    """Numbers stay numbers so columns sort; Forever prices are small (a Wool Cloth is under 2 silver), so gold shows 4 decimals (down to the copper)."""
    cfg = {}
    for c in df.columns:
        if c == "_id" or not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        if c in ("Item ID", "Hours", "Cheapest at", "Dearest at"):
            cfg[c] = st.column_config.NumberColumn(c, format="%d")
        elif c.endswith("(g)"):
            cfg[c] = st.column_config.NumberColumn(c, format="%.4f")
        elif c.endswith("(g/day)"):
            cfg[c] = st.column_config.NumberColumn(c, format="%.1f")
        elif c.endswith("(%)"):
            cfg[c] = st.column_config.NumberColumn(c, format="%.0f%%")
        else:
            cfg[c] = st.column_config.NumberColumn(c, format="%.0f")
    return cfg


def pick_row(df: pd.DataFrame, key: str, wide: dict, height: int = 520):
    ev = st.dataframe(df.drop(columns=["_id"]), **wide, hide_index=True, height=height, key=key,
                      column_config=column_config(df), on_select="rerun", selection_mode="single-row")
    rows = ev.selection.rows if ev is not None and hasattr(ev, "selection") else []
    return int(df["_id"].iloc[rows[0]]) if rows else None


def item_detail(data: dict, house: int, day: str, item_id: int, names: dict, wide: dict, key: str):
    daily, hmap = data["daily"], data["hmap"]
    st.subheader(f"{F.name_of(item_id, names)} ({item_id}) · {F.house_label(house, hmap)}")
    h = F.item_history(daily, house, item_id)
    if h.empty:
        st.info("This item has no stored day in this house.")
        return
    full, part = h[h["complete"]], h[~h["complete"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["day"], y=h["price_p30"], name="cheapest 30% of the stock", line=dict(color="#8A9BA3", dash="dash")))
    fig.add_trace(go.Scatter(x=h["day"], y=h["price_min"], name="lowest seen", line=dict(color="#C28A1B", dash="dot")))
    fig.add_trace(go.Scatter(x=h["day"], y=h["price_day"], name="price (typical lowest)", line=dict(color=INK), mode="lines+markers",
                             marker=dict(size=[9 if c else 6 for c in h["complete"]], symbol=["circle" if c else "circle-open" for c in h["complete"]])))
    fig.update_layout(title="Price per unit by day (gold)", height=340, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0), xaxis=dict(type="category"))
    st.plotly_chart(fig, **wide, key=f"{key}_price")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=h["day"], y=h["sold_low_day"], name="seen sold (lower bound)", marker_color="#8FB3A0"))
    fig2.add_trace(go.Bar(x=h["day"], y=h["sold_cert_day"], name="certainly sold", marker_color="#2E7D4F"))
    fig2.add_trace(go.Scatter(x=h["day"], y=h["units_day"], name="units listed", yaxis="y2", line=dict(color=INK)))
    fig2.update_layout(title="Units sold per day (bars) and units listed (line)", barmode="overlay", height=340, margin=dict(l=10, r=10, t=40, b=10),
                       yaxis2=dict(overlaying="y", side="right", showgrid=False), xaxis=dict(type="category"),
                       legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0))
    st.plotly_chart(fig2, **wide, key=f"{key}_sales")
    st.caption(f"{len(full)} complete day{'s' if len(full) != 1 else ''} and {len(part)} partial (open markers: fewer than {F.COMPLETE_HOURS} usable hours, "
               "usually the first and newest stored days). Sales are lower bounds: Blizzard publishes each house once an hour, so a listing posted and sold inside "
               "the hour is in no snapshot. Use them to compare items and days, not as the true number sold.")
    across = F.item_across_houses(daily, day, item_id, hmap)
    if len(across):
        st.markdown(f"**The same item in every house on {day}** (each house is its own market)")
        st.dataframe(across, hide_index=True, **wide, column_config=column_config(across))
    with st.expander("Every stored day of this item in this house"):
        t = F.history_table(h)
        st.dataframe(t, hide_index=True, **wide, column_config=column_config(t))


def render(wide: dict):
    repo = repo_name()
    data = load_forever(repo)
    st.title("WoW Market Desk · WoW Forever")
    if data["error"]:
        st.error("Could not read the WoW Forever dataset. The app's HF_TOKEN secret needs read access to it (Streamlit: Manage app > Settings > Secrets). "
                 f"Details: {data['error']}")
        st.stop()
    daily, hmap = data["daily"], data["hmap"]
    if daily.empty:
        st.info("No daily tables stored yet. Run the 'Forever daily tables' workflow with save = yes (GitHub > Actions).")
        st.stop()
    names, classes = F.item_names(data["items"]), F.item_classes(data["items"])
    text, stale = F.freshness(daily)
    (st.warning if stale else st.caption)(text)
    if not names:
        st.caption("Item names are not stored yet: run the 'Forever tables' workflow once. Items show as 'Item 12345' until then.")

    houses = F.house_order(daily["house"].unique(), hmap)
    try:
        wanted = int(st.query_params.get("house", F.DEFAULT_HOUSE))
    except (TypeError, ValueError):
        wanted = F.DEFAULT_HOUSE
    c1, c2 = st.columns([2, 2])
    house = c1.selectbox("Auction house", houses, index=houses.index(wanted) if wanted in houses else 0,
                         format_func=lambda h: F.house_label(h, hmap), key="fv_house",
                         help="Forever has no realms: each auction house is its own market. 'Unverified' houses are placed by slot order only.")
    try:
        st.query_params["house"] = str(house)
    except Exception:
        pass
    summ = F.day_summary(daily, house)
    days = list(summ["day"])[::-1]
    dflt = F.default_day(summ)
    day = c2.selectbox("Day (US Eastern)", days, index=days.index(dflt) if dflt in days else 0, format_func=lambda d: F.day_label(d, summ), key="fv_day",
                       help="The newest day is still filling in; the default is the newest day where most items are complete.")

    tab_market, tab_compare, tab_about = st.tabs(["Market", "Compare houses", "About the numbers"])

    with tab_market:
        hd = F.market_headline(daily, house, day)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Items listed", f"{hd['items']:,}", help=f"{hd['complete']:,} with a complete day ({F.COMPLETE_HOURS}+ usable hours)")
        m2.metric("Gold in play", f"{hd['gold']:,.0f}g", help="Price x quantity over every listing, typical hour, complete items only")
        m3.metric("Seen traded per day", f"{hd['traded']:,.0f}g", help="Seen sold x price over complete items. A lower bound: fast sales are invisible.")
        m4.metric("Items seen selling", "-" if hd["selling_share"] != hd["selling_share"] else f"{hd['selling_share']:.0%}")
        labels = F.item_labels(daily.loc[daily["house"] == house, "item_id"].unique(), names)
        by_label = {v: k for k, v in labels.items()}
        f1, f2 = st.columns([3, 2])
        found = f1.selectbox("Find an item (type part of a name or an ID)", sorted(by_label), index=None, placeholder="e.g. 'leather' or 2592", key="fv_find")
        cls_all = F.class_options(daily, house, day, classes)
        cls = f2.multiselect("Item class", cls_all, default=[], key="fv_class", help="Leave empty for every class")
        g1, g2 = st.columns([1, 3])
        min_gold = g1.number_input("Min gold in play", 0.0, 1_000_000.0, 0.0, 10.0, key="fv_min_gold")
        complete = g2.checkbox("Complete days only", value=True, key="fv_complete",
                               help=f"Rows with {F.COMPLETE_HOURS}+ usable hours. Partial rows rest on part of a day.")
        if found:
            item_detail(data, house, day, by_label[found], names, wide, "fv_find")
            st.divider()
        t = F.market_view(daily, house, day, names, classes, min_gold, complete, "", cls or None)
        st.write(f"{len(t):,} items in house {house} on {day}, ranked by gold seen traded per day (seen sold x price). "
                 "Click a row for the item's charts below. Click a column heading to sort by it.")
        sel = pick_row(t, "fv_market", wide) if len(t) else None
        if sel is not None:
            item_detail(data, house, day, sel, names, wide, "fv_market")

    with tab_compare:
        trio = F.same_ruleset(house, hmap, houses)
        st.write("The auction houses of one ruleset side by side: what each item costs in the Alliance, Horde and Neutral (Blackwater) house on the same day. "
                 "A character reaches its own faction's house and the neutral one, so a gap between those two is one you can act on.")
        st.caption("Auction house fees are not included. On Classic-era rules the neutral house takes a bigger cut and deposit than the faction houses; "
                   "that has not been checked on Forever yet, so treat a small gap as no gap.")
        if len(trio) < 2:
            st.info("Only one house of this ruleset has data.")
        else:
            k1, k2 = st.columns([1, 1])
            cmin = k1.number_input(f"Min gold in play in house {house}", 0.0, 1_000_000.0, 0.0, 10.0, key="fv_cmp_gold")
            cun = k2.number_input("Min units listed in a house", 0.0, 100000.0, 20.0, 5.0, key="fv_cmp_units",
                                  help="A house with fewer units than this does not count for the item (a single cheap listing is not a market).")
            ordered = [house] + [h for h in trio if h != house]
            cv = F.compare_view(daily, day, ordered, hmap, names, cmin, True, cun)
            if cv.empty:
                st.info("No item has a complete day in at least two of these houses with these filters.")
            else:
                st.write(f"{len(cv):,} items on {day}, biggest price gap first. Click a row for the item's charts in house {house}.")
                sel2 = pick_row(cv, "fv_compare", wide)
                if sel2 is not None:
                    item_detail(data, house, day, sel2, names, wide, "fv_compare")

    with tab_about:
        st.markdown(
            "**Where these numbers come from.** Every hour the fetcher saves each Forever auction house's commodity listings (reagents, cloth, leather, herbs, "
            "consumables). Once a day the 'Forever daily tables' job boils each house's day down to one row per item. Gear and weapons trade as single "
            "auctions and are not in these tables yet.\n\n"
            "- **Price**: the typical lowest price per unit over the day's usable hours (the median of the hourly lowest prices). **Lowest**: the lowest price seen.\n"
            "- **Cheapest 30%**: the price at which the cheapest 30% of the stock is reached; one cheap listing cannot move it.\n"
            "- **Buy 20 / Buy 200 avg**: the average price per unit to buy 20 or 200 units, walking up the listings. Blank when that many are not usually for sale.\n"
            "- **Units listed**, **Gold in play**: the typical hour's stock and its value.\n"
            "- **Seen sold/day**: the conservative estimate of units sold. **Certainly sold/day**: only sales that are certain (a listing still standing with fewer units).\n"
            "- **Turnover**: seen sold / units listed. **Hours selling**: the share of the day's hours in which any sale was seen.\n"
            "- **Hours**: usable hourly readings behind the row; 18 or more is a complete day.\n\n"
            "**Every sales number is a lower bound.** Blizzard publishes each house once an hour. A listing posted and bought inside that hour is in no "
            "snapshot, for this tool or any other. On Forever many sales happen within minutes, so a large part of the volume is invisible. The sales "
            "columns are for comparing items and days, not for the true number. Measuring how much the snapshots miss needs your own sales and posting times.\n\n"
            "**Houses.** " + " ".join(f"{F.house_label(h, hmap)}." for h in houses))
