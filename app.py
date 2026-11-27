"""WoW market desk, on WoW FOREVER data (v0.7.50). Reads the small tables the Forever derive job writes to engine/h<house>/ in the Forever dataset on Hugging Face.
Pick a ruleset, then an auction house; every tab shows that house (houses are never added together).

Every estimate is shown as low - best - high with a confidence label. Nothing here is advice to buy; the ranges are the point.
Click a row in the Market or Flips table to see that item's charts below it."""
import os

import houseflip as HFL
import forever_lib as F
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lib

st.set_page_config(page_title="WoW Market Desk · WoW Forever", layout="wide")


def _secret(name):
    try:
        v = st.secrets.get(name)
    except Exception:
        v = None
    return v or os.environ.get(name)


HF_REPO = _secret("FOREVER_HF_REPO") or F.DEFAULT_REPO        # the Forever dataset (the retail dataset is no longer read)


def _stretch() -> dict:
    """Streamlit 1.50+ wants width="stretch"; older versions only know use_container_width. Use whichever this server has."""
    try:
        v = tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        v = (0, 0)
    return {"width": "stretch"} if v >= (1, 50) else {"use_container_width": True}


WIDE = _stretch()


@st.cache_data(ttl=600, show_spinner="Loading tables...")
def load(name: str, house: int) -> pd.DataFrame:
    """One table of one house: engine/h<house>/<name>.parquet in the Forever dataset."""
    local = os.environ.get("DERIVED_DIR")
    try:
        if local:
            return pd.read_parquet(os.path.join(local, f"h{int(house)}", f"{name}.parquet"))
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HF_REPO, f"engine/h{int(house)}/{name}.parquet", repo_type="dataset", token=_secret("HF_TOKEN"))
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_houses() -> tuple:
    """(houses with tables, {house: ruleset / faction / confidence}) from engine/houses.json; every non-empty house, unlabelled, if it is missing."""
    local = os.environ.get("DERIVED_DIR")
    data = None
    try:
        if local:
            with open(os.path.join(local, "houses.json"), "rb") as fh:
                data = fh.read()
        else:
            from huggingface_hub import hf_hub_download
            with open(hf_hub_download(HF_REPO, "engine/houses.json", repo_type="dataset", token=_secret("HF_TOKEN")), "rb") as fh:
                data = fh.read()
    except Exception:
        data = None
    built, labels, _ = F.parse_houses_file(data)
    hmap = F.parse_house_map(None)
    hmap.update({h: {**hmap.get(h, {}), **{k: v for k, v in r.items() if v}} for h, r in labels.items()})
    return (built or sorted(hmap)), hmap


# ---------------------------------------------------------------- Ruleset, then auction house (kept in the page address: ?ruleset=pvp&house=110)
HOUSES, HMAP = load_houses()
RULESETS = F.rulesets_of(HOUSES, HMAP)
_q_house = lib.parse_ids(str(st.query_params.get("house", "")))
_q_house = _q_house[0] if _q_house else 110
_q_rs = str(st.query_params.get("ruleset", "")) or (HMAP.get(_q_house) or {}).get("ruleset", "pvp")
p1, p2, _ = st.columns([1, 1, 2])
RULESET = p1.selectbox("Ruleset", RULESETS, index=RULESETS.index(_q_rs) if _q_rs in RULESETS else 0, format_func=lambda r: F.RULESET_NAME.get(r, r), key="pick_ruleset")
HOUSE_OPTS = F.houses_of(RULESET, HOUSES, HMAP)
HOUSE = p2.selectbox("Auction house", HOUSE_OPTS, index=HOUSE_OPTS.index(_q_house) if _q_house in HOUSE_OPTS else 0,
                     format_func=lambda h: F.picker_label(h, HMAP), key="pick_house",
                     help="WoW Forever has no realms: each auction house is its own market. '(unverified)' = placed by slot order only, not yet confirmed by a listing test.")
try:
    st.query_params["ruleset"] = RULESET
    st.query_params["house"] = str(HOUSE)
except Exception:
    pass

items, hourly, prof = load("items", HOUSE), load("item_hourly", HOUSE), load("profile", HOUSE)
cal, hist, flips = load("calibration", HOUSE), load("calibration_history", HOUSE), load("flips", HOUSE)
shelves = load("shelves", HOUSE)
ledger = lib.clean_ledger(load("forecast_ledger", HOUSE))      # forecasts are not built for Forever yet (empty)
shelves_old = len(shelves) > 0 and "test_regime" not in shelves.columns
shelves_ver = str(shelves["engine_version"].iloc[0]) if len(shelves) and "engine_version" in shelves.columns else "unknown"
shelves = lib.ensure_shelf_columns(shelves)
backtest = load("backtest_summary", HOUSE)
bt_rates = lib.backtest_rates(backtest)
buy_targets, buy_cal, buy_when, buy_meta = (load(n, HOUSE) for n in ("buy_targets", "buy_calibration", "buy_when", "buy_meta"))   # not built for Forever yet (empty)
if hist.empty:
    hist = cal

st.title("WoW Market Desk · WoW Forever")
hrs, state = lib.freshness(cal)
if items.empty:
    st.error(f"No tables for house {HOUSE} yet. Run the 'Forever derive' workflow with save = yes (GitHub > Actions). If it has run, the app's HF_TOKEN secret "
             "may not be able to read the Forever dataset (Streamlit: Manage app > Settings > Secrets).")
    st.stop()
ver = items["engine_version"].iloc[0] if "engine_version" in items else "?"
st.caption(f"House {F.picker_label(HOUSE, HMAP)} ({F.RULESET_NAME.get(RULESET, RULESET)}). Data {state}: built from snapshots up to {pd.Timestamp(cal['data_last_ts'].max()):%a %b %d %H:%M} UTC "
           f"({hrs:.1f} h ago) | engine {ver} | ranges are low - best - high; labels say how much to trust them")


def column_config(df: pd.DataFrame) -> dict:
    """Numbers stay numbers (so the column sorts properly) and are only formatted for display. Gold shows 4 decimals (one copper): Forever prices are small."""
    cfg = {}
    for c in df.columns:
        if c == "_id" or not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        if c == "Item ID":
            cfg[c] = st.column_config.NumberColumn(c, format="%d")
        elif c.endswith("(g)"):
            cfg[c] = st.column_config.NumberColumn(c, format="%.4f")
        elif c.endswith("(%)"):
            cfg[c] = st.column_config.NumberColumn(c, format="%.1f%%")
        elif c.endswith(("(units)", "passed", "failed", "warned")):
            cfg[c] = st.column_config.NumberColumn(c, format="%d")
        else:
            cfg[c] = st.column_config.NumberColumn(c, format="%.1f")
    if "Watched" in df.columns:
        cfg["Watched"] = st.column_config.CheckboxColumn("Watched")
    return cfg


def score_config(df: pd.DataFrame) -> dict:
    """Column formats for the forecast scoreboards: counts are whole numbers, the 'closer than staying put' ratio shows two decimals."""
    cfg = column_config(df)
    for c in ("Cases", "Items"):
        if c in df.columns:
            cfg[c] = st.column_config.NumberColumn(c, format="%d")
    if "Closer than staying put?" in df.columns:
        cfg["Closer than staying put?"] = st.column_config.NumberColumn("Closer than staying put?", format="%.2f")
    return cfg


def pick_row(df: pd.DataFrame, key: str, height: int = 520):
    """Show a table where one row can be clicked. Returns the item_id of the clicked row, or None."""
    ev = st.dataframe(df.drop(columns=["_id"]), **WIDE, hide_index=True, height=height, key=key,
                      column_config=column_config(df), on_select="rerun", selection_mode="single-row")
    rows = ev.selection.rows if ev is not None and hasattr(ev, "selection") else []
    return int(df["_id"].iloc[rows[0]]) if rows else None


def time_charts(h: pd.DataFrame, key: str):
    """Three heatmaps, day of week x hour of day (US Eastern): quantity listed, lowest price, sales velocity."""
    tp = lib.time_profiles(h)
    if not tp:
        return
    days = (h["ts"].max() - h["ts"].min()).total_seconds() / 86400
    st.markdown("**When it happens: day of week and hour of day (US Eastern)**")
    st.caption(f"Built from {tp['hours']} hourly readings ({days:.1f} days). Each square is the median of the readings in that hour of that weekday; "
               "with under a month of data most squares rest on one to three readings, so they are a draft. Blank squares have no reading. "
               "Hover a square for its value and how many readings are behind it.")
    n = tp["n"]
    specs = [
        ("units", "Quantity listed (total units on the auction house)", "YlOrBr", lambda v: f"{v:,.0f} units"),
        ("price", "Lowest price", "Blues", lib.gsc),
        ("sold", "Sales velocity (best estimate of units sold per hour)", "Greens", lambda v: f"{v:,.1f} per hour"),
    ]
    for gk, title, scale, fmt in specs:
        z = tp[gk]
        text = z.apply(lambda col: col.map(lambda v: "" if v != v else fmt(v)))
        hover = text.copy()
        for r in range(7):
            for c in range(24):
                hover.iat[r, c] = f"{lib.DOW[r]} {lib.HOURS[c]}: {text.iat[r, c]} ({n.iat[r, c]} reading{'s' if n.iat[r, c] != 1 else ''})"
        f = go.Figure(go.Heatmap(z=z.values, x=lib.HOURS, y=lib.DOW, colorscale=scale, xgap=2, ygap=2, hoverongaps=False,
                                 text=hover.values, hovertemplate="%{text}<extra></extra>", colorbar=dict(thickness=10)))
        f.update_layout(title=title, height=300, margin=dict(l=10, r=10, t=50, b=10), yaxis=dict(autorange="reversed"),
                        xaxis=dict(type="category", tickangle=0, tickmode="array", tickvals=lib.HOURS[::2], ticktext=lib.HOURS[::2]))
        st.plotly_chart(f, **WIDE, key=f"{key}_heat_{gk}")


def item_detail(pick: int, key: str):
    """Everything we know about one item, shown under whichever table it was clicked in."""
    r = items[items["item_id"] == pick]
    if r.empty:
        st.info("That item is not in the tables.")
        return
    r = r.iloc[0]
    title = r["name"] if r["name"] else f"item {pick}"
    st.subheader(f"{title}  (item ID {pick})")
    k = st.columns(4)
    k[0].metric("Lowest price now", lib.gsc(r["floor_gold"]), lib.price_position(r))
    k[1].metric("Typical band (p10 - p90)", f"{lib.gsc(r['band_p10'])} - {lib.gsc(r['band_p90'])}")
    k[2].metric("Sold per day (low - best - high)", lib.rng(r["daily_low"], r["daily_est"], r["daily_high"]))
    k[3].markdown(f"**Confidence: <span style='color:{lib.CONF_COLOR[r['confidence']]}'>{r['confidence']}</span>**<br>{r['confidence_reason']}",
                  unsafe_allow_html=True)
    h = hourly[hourly["item_id"] == pick].sort_values("ts").reset_index(drop=True) if "item_id" in hourly else pd.DataFrame()
    if h.empty:
        st.info("No hourly history for this item in the tables yet.")
        return
    methods = lib.item_methods(ledger, pick)
    fc_method = None
    if methods:
        fc_method = st.selectbox("Forecast drawn on the chart", methods, format_func=lambda m: lib.METHOD_NAME.get(m, m), key=f"{key}_fc_method")
        if fc_method not in methods:
            fc_method = methods[0]
    fig = go.Figure()
    # Constant lines are drawn as traces (not hline / hrect) so hovering anywhere shows their numbers, same for every point.
    g = lambda v: f"{v:,.4f}g"
    fig.add_trace(go.Scatter(x=h["ts"], y=[r["band_p90"]] * len(h), line=dict(width=0), name="typical high (p90)",
                             hovertemplate="%{y:,.4f}g", showlegend=False))
    fig.add_trace(go.Scatter(x=h["ts"], y=[r["band_p10"]] * len(h), line=dict(width=0), fill="tonexty", fillcolor="rgba(43,76,89,0.12)",
                             name=f"typical band (p10 {g(r['band_p10'])} - p90 {g(r['band_p90'])})", hovertemplate="%{y:,.4f}g"))
    fig.add_trace(go.Scatter(x=h["ts"], y=h["floor_gold"], name="lowest price", line=dict(color="#2B4C59"), hovertemplate="%{y:,.4f}g"))
    sp = h.loc[lib.flag_mask(h, "spike")]
    fig.add_trace(go.Scatter(x=sp["ts"], y=sp["floor_gold"], mode="markers", name="price spike", marker=dict(color="#B5483A", size=8),
                             hovertemplate="%{y:,.4f}g"))
    bl = h.loc[lib.flag_mask(h, "blip")]
    fig.add_trace(go.Scatter(x=bl["ts"], y=bl["floor_gold"], mode="markers", name="listing blip (likely API)",
                             marker=dict(color="#C28A1B", symbol="x", size=8), hovertemplate="%{y:,.4f}g"))
    for col, nm, first in (("wall_low_gold", "wall low", True), ("wall_high_gold", "wall high", False)):
        if col in r and r[col] == r[col]:
            fig.add_trace(go.Scatter(x=h["ts"], y=[r[col]] * len(h), name="price wall (low / high)", legendgroup="wall", showlegend=first,
                                     line=dict(dash="dot", color="#8A9BA3"), hovertemplate=nm + " %{y:,.4f}g"))
    fc = lib.item_forecast(ledger, pick, fc_method) if fc_method else pd.DataFrame()
    dp = lib.item_daily_points(ledger, pick)
    if len(dp):
        fig.add_trace(go.Scatter(x=[lib.day_to_utc(d) for d in dp["day"]], y=dp["price"], mode="markers", name="daily price (median of the day)",
                                 marker=dict(color="#1B2A33", symbol="diamond", size=9), hovertemplate="%{y:,.4f}g"))
    if len(fc) and fc["origin"].iloc[0] == fc["origin"].iloc[0]:
        o, made = float(fc["origin"].iloc[0]), fc["made_day"].iloc[0]
        X = [lib.day_to_utc(made)] + [lib.day_to_utc(d) for d in fc["target_day"]]
        for lo, hi, alpha, nm in (("lo80", "hi80", 0.12, "80% range"), ("lo50", "hi50", 0.26, "50% range")):
            fig.add_trace(go.Scatter(x=X + X[::-1], y=[o] + list(fc[hi]) + list(fc[lo])[::-1] + [o], mode="lines", fill="toself", line=dict(width=0),
                                     fillcolor=f"rgba(58,110,165,{alpha})", name=f"forecast {nm}", hoverinfo="skip"))
        tips = [f"daily price on {made}: {o:,.4f}g"] + [
            f"{lib.HORIZON_NAME.get(int(r['horizon']), str(r['horizon']))} ahead ({r['target_day']}): guess {r['guess']:,.2f}g, "
            f"50% range {r['lo50']:,.2f} - {r['hi50']:,.2f}g, 80% range {r['lo80']:,.2f} - {r['hi80']:,.2f}g" for _, r in fc.iterrows()]
        fig.add_trace(go.Scatter(x=X, y=[o] + list(fc["guess"]), mode="lines+markers", name=f"forecast: {lib.METHOD_NAME.get(fc_method, fc_method)}",
                                 line=dict(color=lib.FORECAST_COLOR, dash="dot"), marker=dict(size=8), text=tips, hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(title=f"Lowest price per hour (UTC), {len(h)} hours of history", height=420, hovermode="x unified",
                      margin=dict(l=10, r=10, t=40, b=10), yaxis_title="gold",
                      legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0))
    st.plotly_chart(fig, **WIDE, key=f"{key}_price")
    if len(fc):
        st.caption("Forecast: the dotted line is the guess for the item's daily price (the median of that US Eastern day's hourly lowest prices; the diamonds are those daily prices). "
                   "The dark and light blue bands are the 50% and 80% ranges: the real daily price should land inside them about half and four times in five. "
                   "A forecast is a record, not advice. The Forecasts tab says whether the ranges have held.")
    elif len(ledger):
        st.caption("No forecast for this item yet. It needs complete days of our own prices first: 5 for 'Stays put', 7 for 'Recent average', 12 for 'Pullback model'.")
    past = lib.item_history_table(ledger, pick)
    if len(past):
        with st.expander(f"Forecasts made for this item so far ({len(past)} newest shown)"):
            st.dataframe(past, hide_index=True, **WIDE, column_config=column_config(past))

    rb = lib.daily_ribbon(h)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=rb["ts"], y=rb["high"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=rb["ts"], y=rb["low"], fill="tonexty", fillcolor="rgba(43,76,89,0.18)", line=dict(width=0), name="low - high"))
    fig2.add_trace(go.Scatter(x=rb["ts"], y=rb["est"], name="best estimate", line=dict(color="#2B4C59")))
    fig2.update_layout(title="Units sold in the last 24 hours (rolling), as a range", height=340, margin=dict(l=10, r=10, t=40, b=10),
                       legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0))
    st.plotly_chart(fig2, **WIDE, key=f"{key}_sales")
    st.caption(f"{len(lib.hole_hours(h))} hours have no estimate (a hole in our data or a flagged interval). The shaded band is the range; "
               "it is not a statistical confidence interval, it is the spread between the cautious and the generous reading of the same data.")

    if len(shelves):
        s1 = shelves[shelves["item_id"] == pick]
        if len(s1):
            x = s1.iloc[0]
            st.markdown(f"**Flip shelf: {x['verdict']}.** Buy {x['units']:.0f} units (avg {lib.gsc(x['avg_price_gold'])}, break-even "
                        f"{lib.gsc(x['breakeven_gold'])}), relist at {lib.gsc(x['target_gold'])}: net margin {x['net_margin']:.0%}, "
                        f"{x['net_margin_stress']:.0%} if you sell 10% under. Hours to sell {lib.shelf_hours(x)}.")
            st.caption(f"Checks that did not pass: {lib.shelf_watch(x)}. What each check means is explained on the Flips tab, under its table.")
    time_charts(h, key)
    if len(flips):
        f = flips[flips["item_id"] == pick]
        if len(f):
            st.markdown("**Past buyouts of this item's cheap listings**")
            st.caption("Moments in the last 4 days when almost all of the cheapest listings vanished within one hour (bought out, not expired), "
                       "so the lowest price jumped. It may have been a flipper, or just a rush of demand. 'Back to old price?' says how long it took "
                       "for the lowest price to return to within 10% of what it was before. 'Looks like a flipper' means the amount bought was at "
                       "least 3 times a normal hour of sales AND at least half of it was relisted near the new price. Use this to see how long a "
                       "raised price tends to last for this item.")
            ev = lib.past_events_table(f)
            st.dataframe(ev, hide_index=True, **WIDE, column_config=column_config(ev))

tab_market, tab_buy, tab_flip, tab_house, tab_fc, tab_conf = st.tabs(["Market", "Buy targets", "Flips", "House flips", "Forecasts", "Confidence"])

# ---------------------------------------------------------------- Market
with tab_market:
    labels = lib.item_labels(items)                              # {item_id: "Name (id)"}
    by_label = {v: k for k, v in labels.items()}
    options = sorted(by_label)

    # Watch list: kept in the page address (?watch=1,2,3), so bookmark the page and it comes back.
    if "watch_sel" not in st.session_state:
        saved = lib.parse_ids(st.query_params.get("watch", ""))
        st.session_state["watch_sel"] = [labels[i] for i in saved if i in labels]
    c1, c2 = st.columns([3, 3])
    found = c1.selectbox("Find an item (type part of a name or an ID)", options, index=None, placeholder="e.g. 'saronite' or 'Saron' or 224828",
                         key="find_item")
    with c2:
        watch_labels = st.multiselect("Watch list (type to add items)", options, key="watch_sel",
                                      help="Your own list. It is saved in this page's web address: bookmark the page to keep it.")
    watch_ids = [by_label[x] for x in watch_labels]
    try:
        st.query_params["watch"] = ",".join(str(i) for i in watch_ids)
    except Exception:
        pass
    c3, c4, c5 = st.columns([1, 3, 1])
    minf = c3.number_input("Min price (gold)", 0.0, 100000.0, 0.0, 0.5)
    conf = c4.multiselect("Confidence", lib.CONF_ORDER, default=lib.CONF_ORDER)
    watched = c5.checkbox("Watched only")
    if found:
        item_detail(by_label[found], "find")
        st.divider()
    if watched and not watch_ids:
        st.info("Your watch list is empty. Add items in the Watch list box above.")
    t = lib.market_table(items, minf, tuple(conf), "", watched, watch_ids=watch_ids)
    st.write(f"{len(t):,} items, ranked by estimated gold flow per day (best estimate). The low / best / high columns are the honest part. "
             "Click a row to see the item's charts below. Click a column heading to sort by it.")
    sel = None
    if len(t):
        sel = pick_row(lib.market_numbers(t), "market_table")
    if sel is not None:
        item_detail(sel, "market")

# ---------------------------------------------------------------- Flips
# ---------------------------------------------------------------- Buy targets
with tab_buy:
    st.write("What to pay when you are gathering inventory. For every item, the price at or under which it is a good buy compared with its own last week, "
             "and how often that price has actually shown up. Scan the auction house in game and buy what is at or under the number. "
             "This does not look at what you will do with the items.")
    if buy_targets.empty:
        st.info("Buy targets are not built for WoW Forever yet: they need the hourly history per house and the buy-window job (the next step after this version). "
                "The tab will fill in by itself once that job runs.")
    else:
        caption, stale = lib.buy_meta_caption(buy_meta)
        if caption:
            (st.warning if stale else st.caption)(caption)
        bt = lib.buy_names(buy_targets, items)
        c1, c2, c3 = st.columns([2, 2, 2])
        tier_key = c1.radio("How low", list(lib.BUY_TIERS), index=1, format_func=lambda k: lib.BUY_TIERS[k], key="buy_tier",
                            help="Good prices come up often; steals are rare but the cheapest. The chance columns say how often.")
        tiers_present = sorted(bt["price_tier"].dropna().unique().tolist()) if "price_tier" in bt.columns else []
        price_tiers = c2.multiselect("Price level", tiers_present, default=tiers_present, key="buy_price_tiers",
                                     help="Items are judged against items of a similar price, so each level has its own calibration.")
        group_opts = sorted(g for g in bt["group"].dropna().unique().tolist() if g) if "group" in bt.columns else []
        groups_sel = c3.multiselect("Item group", group_opts, key="buy_groups", help="Leave empty for all groups.")
        show_extreme = st.checkbox("Include extreme discounts (target more than 50% under the median)", value=False, key="buy_extreme",
                                   help="Left out by default. A price that far under an item's own week is usually a one-off listing or an item whose price swings between two levels. "
                                        "Tick to see them anyway; each carries a note.")
        d1, d2, d3, d4 = st.columns(4)
        query = d1.text_input("Name or ID contains", key="buy_query")
        min_prob = d2.slider("At least this chance within 3 days (%)", 0, 100, 50, 5, key="buy_min_prob") / 100.0
        min_depth = d3.slider("At least this far under the median (%)", 0, 30, 3, 1, key="buy_min_depth") / 100.0
        min_units = d4.number_input("At least this many units near the floor", min_value=0, value=0, step=10, key="buy_min_units")
        view_args = dict(tier=tier_key, price_tiers=price_tiers, groups=groups_sel, query=query, min_depth=min_depth, min_prob=min_prob, prob_days=3, min_units=float(min_units))
        view = lib.buy_view(bt, max_depth=None if show_extreme else lib.BUY_EXTREME, **view_args)
        hidden = 0 if show_extreme else len(lib.buy_view(bt, max_depth=None, **view_args)) - len(view)
        a1, a2, a3 = st.columns(3)
        a1.metric("Items on this list", f"{len(view):,}")
        a2.metric("Combined worth of the list", lib.short_num(float(view["Worth (g)"].sum())) + "g" if len(view) else "0g",
                  help="Each item's Worth added up: units usually sitting near the floor at that price, times how far the target is under the 7-day median. It assumes you bought every "
                       "item's whole usual pile, so read it as a size gauge, not a total you could realistically save. It is a saving against paying the median, not a profit.")
        rhythm = lib.buy_when_verdict(lib.buy_when_row(buy_when, "everything", tier_key))[1] == "rhythm"
        a3.metric("Best hours (US Eastern)", "see below" if rhythm else "no reliable hour", help="Whether some hours of the day reliably have more low prices. Details at the bottom of this tab.")
        if hidden:
            st.caption(f"{hidden:,} more item{'s' if hidden != 1 else ''} would pass these filters but {'has' if hidden == 1 else 'have'} a target more than 50% under {'its' if hidden == 1 else 'their'} median, "
                       "so they are left out. Tick the box above to see them.")
        if view.empty:
            st.info("No item passes these filters. Loosen the chance or depth sliders.")
        else:
            sel = pick_row(view, "buy_table", height=520)
            with st.expander("Copy the list for your in-game notes (best first)"):
                st.code(lib.buy_copy_text(view, 40), language=None)
                st.download_button("Download the whole filtered list (CSV)", view.drop(columns=["_id"]).to_csv(index=False, float_format="%.4f"),
                                   file_name="buy_targets.csv", mime="text/csv", key="buy_csv")
            if sel is not None:
                if sel in set(items["item_id"].astype(int)):
                    item_detail(sel, "buy")
                else:
                    st.caption("This item has no market-table row yet, so there are no item charts for it.")
        with st.expander("How reliable are these prices?"):
            st.write("We replayed history: for each item and day, set the target from the previous 7 days only, then watched whether the price reached it on that day "
                     "and on the next days. Rows are how far under the median the target was. These are days the tool had not seen when it set the price.")
            pools = sorted(buy_cal["pool"].unique().tolist()) if not buy_cal.empty else []
            if pools:
                pool = st.selectbox("Judged over", pools, index=pools.index("everything") if "everything" in pools else 0, key="buy_pool")
                cv = lib.buy_calibration_view(buy_cal, pool, tier_key)
                st.dataframe(cv, hide_index=True, **WIDE, column_config=column_config(cv))
                st.caption("The deeper the target, the less often it is reached and the fewer units are there when it is. 'Saving vs average price' is how much under the "
                           "average hourly price the first reach was. A wait is how many hours into the day it first showed up.")
            else:
                st.caption("Not enough graded days yet.")
        with st.expander("When is it cheapest? (hour of day, US Eastern)"):
            wp = sorted(buy_when["pool"].unique().tolist()) if not buy_when.empty else []
            if not wp:
                st.caption("Not enough days yet to look at hours (it needs about 8 days of data).")
            else:
                wpool = st.selectbox("Judged over", wp, index=wp.index("everything") if "everything" in wp else 0, key="buy_when_pool")
                row = lib.buy_when_row(buy_when, wpool, tier_key)
                words, level = lib.buy_when_verdict(row, int(buy_meta["window_hours"].iloc[0]) if not buy_meta.empty else 3)
                (st.success if level == "rhythm" else st.info)(words)
                lifts = lib.buy_when_lifts(row)
                if lifts:
                    f = go.Figure(go.Bar(x=lib.HOURS, y=lifts, marker_color=["#2E7D4F" if (v == v and v >= 1.15) else "#9AA5AB" for v in lifts]))
                    f.add_hline(y=1.0, line_dash="dot")
                    f.update_layout(title="Chance of reaching the target, by hour, as a multiple of the average (1.0 = same as any hour)", height=300,
                                    margin=dict(l=10, r=10, t=50, b=10), xaxis=dict(type="category"))
                    st.plotly_chart(f, **WIDE, key="buy_when_chart")
                st.caption("The verdict above is based on days the hour pattern was not learned from. A green bar alone means little: with only a few weeks of days, "
                           "some hours always look better by chance.")
        st.caption("These are price levels, not instructions. They are set from the item's own recent past, so after a real shift (a patch, a new season) they lag for about a week. "
                   "Prices shown on the Market tab can be older than these.")

with tab_flip:
    st.write("Cheap units sitting below a gap in the price tiers: buy them, relist at the tier above the gap. These are candidates to "
             "check against the live auction house, not instructions. Limits per flip come from the engine run (units and gold). "
             "Click a row to see the item's charts and its full list of checks below.")
    if shelves_old:
        st.warning(f"This flip table was written by engine {shelves_ver}, but the other tables by engine {ver}, so the normal-price check and "
                   "sales-per-day columns are blank. If you just ran derive: check that its log says 'Uploaded ... files to engine/', then use the menu "
                   "(top right) > Clear cache, or wait 10 minutes. If the engine numbers are old, the engine code in the repo needs updating first.")
    only = st.checkbox("Candidates only (hide weak)", value=False)
    sh = lib.shelf_table(shelves, ("candidate",) if only else ("candidate", "weak"))
    sel = None
    if sh.empty:
        st.info("No shelves pass right now. That is normal; they are rare.")
    else:
        show = lib.flip_numbers(sh, bt_rates)
        sel = pick_row(show, "flip_table")
    with st.expander("What the checks mean"):
        st.markdown("Each shelf is tested six ways. **Verdict: candidate** means no check failed and at most one warned; **weak** means none failed "
                    "but more than one warned. The 'Watch out for' column lists anything that did not pass.")
        st.table(pd.DataFrame({"Check": [lib.TEST_INFO[c][0] for c in lib.TEST_COLS],
                               "What it asks": [lib.TEST_INFO[c][1] for c in lib.TEST_COLS],
                               "A WARN or FAIL means": [lib.TEST_INFO[c][2] for c in lib.TEST_COLS]}).set_index("Check"))
        st.caption("None of these know whether someone else will undercut you, or whether the cheap units belong to another flipper.")
    if sel is not None:
        item_detail(sel, "flip")
    if not backtest.empty:
        with st.expander("How the screener has done on past hours (backtest)", expanded=False):
            st.write("We replayed the screener on past snapshots, using only what was known at each moment, and looked at what happened next. "
                     "Each shelf is counted once, the first time it appeared. Compare the verdict rows and the pass / warn / fail rows: "
                     "a check is useful if shelves that pass do better than shelves that fail.")
            st.dataframe(lib.backtest_table(backtest), hide_index=True, **WIDE)
            st.caption("The cheap units stay in the book in the replay, so 'reached' is a cautious reading. Undercut risk is the better "
                       "read of whether cheaper listings would have appeared. This is price behaviour, not a record of profit.")

# ---------------------------------------------------------------- House flips (buy on one house of the ruleset, resell on another)
with tab_house:
    faction = (HMAP.get(HOUSE) or {}).get("faction")
    neutral = F.neutral_of(HOUSE, HMAP, HOUSES)
    if faction not in ("alliance", "horde"):
        st.info("Pick your Alliance or Horde house above: this tab compares it with the neutral (Blackwater) house of the same ruleset.")
    elif neutral is None:
        st.info("The neutral house of this ruleset has no tables yet, so there is nothing to compare with.")
    else:
        mine, other = F.picker_label(HOUSE, HMAP), F.picker_label(neutral, HMAP)
        fname = F.HOUSE_NAME[faction]
        direction = st.radio("Direction", ["n2f", "f2n"], horizontal=True, key="hf_direction",
                             format_func=lambda d: f"Buy on Blackwater (neutral), resell on {fname}" if d == "n2f" else f"Buy on {fname}, resell on Blackwater (neutral)")
        h1, h2, h3 = st.columns(3)
        if direction == "n2f":
            need = h1.slider("Margin after the cut, at least (%)", 0, 100, int(HFL.DEFAULTS["min_margin"] * 100), 5, key="hf_margin") / 100.0
            fast_share, fast_margin = None, None
        else:
            need = h1.slider("Margin after the cut, at least (%)", 0, 150, int(HFL.DEFAULTS["min_margin_reverse"] * 100), 5, key="hf_margin_rev",
                             help="The neutral house sells slowly, so the reverse direction asks for a bigger margin...") / 100.0
            fast_share = h2.slider("...unless seen selling on Blackwater in at least this share of hours (%)", 0, 100, int(HFL.DEFAULTS["fast_active_share"] * 100), 5,
                                   key="hf_fast_share") / 100.0
            fast_margin = HFL.DEFAULTS["min_margin"]
        days = h3.slider("Buy no more than the resell house sells in (days)", 1, 7, int(HFL.DEFAULTS["days_to_sell"]), 1, key="hf_days")
        buy_house, sell_house = (neutral, HOUSE) if direction == "n2f" else (HOUSE, neutral)
        cut = HFL.FEES["faction_cut"] if sell_house == HOUSE else HFL.FEES["neutral_cut"]
        book = load("book_now", buy_house)
        sell_items = items if sell_house == HOUSE else load("items", sell_house)
        names = {int(i): n for i, n in zip(items["item_id"], items["name"]) if n}
        hf = HFL.flip_table(book, sell_items, cut, need, days, names, fast_share, fast_margin)
        when = pd.Timestamp(book["ts"].max()).strftime("%a %b %d %H:%M") + " UTC" if len(book) and "ts" in book.columns else "unknown"
        st.write(f"Items listed on **{F.picker_label(buy_house, HMAP)}** that can be resold on **{F.picker_label(sell_house, HMAP)}**, biggest profit first. "
                 f"Listings as of {when}. Click a row for the item's charts on your house.")
        if book.empty:
            st.info(f"No order book stored for house {buy_house} yet (the Forever derive job writes it).")
        elif hf.empty:
            st.info("Nothing passes right now with these settings.")
        else:
            a1, a2, a3 = st.columns(3)
            a1.metric("Items", f"{len(hf):,}")
            a2.metric("Profit after cut, all rows", f"{hf['Profit after cut (g)'].sum():,.2f}g", help="If you bought every row's units and sold them all at the resell price.")
            a3.metric("Gold needed, all rows", f"{hf['Cost (g)'].sum():,.2f}g")
            sel_h = pick_row(hf, "house_flip_table")
            if sel_h is not None:
                if sel_h in set(items["item_id"].astype(int)):
                    item_detail(sel_h, "houseflip")
                else:
                    st.caption("This item has no market row on your house, so there are no charts for it here.")
        st.caption("Profit and margin already have the resell house's auction house cut taken off (the assumptions are in the note at the bottom of the page). "
                   "Sales per day are lower bounds, so 'days to sell' is cautious.")

# ---------------------------------------------------------------- Forecasts
with tab_fc:
    cnt = lib.ledger_counts(ledger)
    if not cnt["made"]:
        st.info("Forecasts are not built for WoW Forever yet. Their ranges were sized from retail price history, which does not exist for Forever; they will be sized "
                "from our own Forever daily history once there are enough days. Once there are some, click an item on the Market tab to see its forecast on the price chart.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Forecasts recorded", f"{cnt['made']:,}")
        m2.metric("Items forecast", f"{cnt['items']:,}")
        m3.metric("Graded so far", f"{cnt['graded']:,}")
        m4.metric("Waiting for their day", f"{cnt['waiting']:,}")
        st.caption(f"Every forecast is written down with its ranges BEFORE the day happens, then graded against the real daily price (the median of the day's hourly lowest prices, "
                   f"US Eastern days). Guesses were made from days {cnt['first_origin']} to {cnt['last_origin']}"
                   + (f"; the latest day graded so far is {cnt['last_target']}." if cnt["last_target"] else "."))
        board = lib.ledger_scoreboard(ledger)
        if not board["graded"]:
            st.info("Nothing is graded yet. A forecast is graded once its day is complete (18 or more usable hours): 1-day forecasts the morning after, 3-day forecasts after three days, "
                    "7-day forecasts after a week. The scoreboard fills in by itself as the daily workflow runs.")
            wt = board["waiting"]
            if len(wt):
                wdf = pd.DataFrame({"Looking ahead": [lib.HORIZON_NAME.get(int(h), f"{int(h)} days") for h in wt["horizon"]], "Forecasts waiting": wt["rows"].astype(int).to_numpy(),
                                    "Items": wt["items"].astype(int).to_numpy(), "First day due": wt["first_target"].to_numpy(), "Last day due": wt["last_target"].to_numpy()})
                st.dataframe(wdf, hide_index=True, **WIDE, column_config={c: st.column_config.NumberColumn(c, format="%d") for c in ("Forecasts waiting", "Items")})
        else:
            ovr = board["overall"]
            ov = lib.score_table(ovr)
            st.markdown("**Scoreboard: every graded forecast so far**")
            st.dataframe(ov, hide_index=True, **WIDE, column_config=score_config(ov))
            st.caption("Typical miss: how far the real price was from the guess, in percent (the middle case; half the cases missed by less). Closer than staying put?: below 1.00 means "
                       "that method's guesses were closer than simply assuming the price stays where it was, over exactly the same cases; 'Stays put' is 1.00 by definition. "
                       "Inside 50% / 80% range: should be near 50 and 80. Much lower means the ranges are too narrow, much higher means they are wider than they need to be. "
                       f"A row with under {lib.MIN_TRUST_CASES} cases is a first look, not a verdict. The ranges were learned from retail history, so this table is also the check "
                       "on whether that history carries over to our own prices.")
            hz_all = [int(h) for h in sorted(ovr["horizon"].unique())]
            hname = lambda h: lib.HORIZON_NAME.get(int(h), f"{int(h)} days")

            def bars(col, title, target):
                f = go.Figure()
                for m in lib.METHODS:
                    s1 = ovr[ovr["method"] == m]
                    if s1.empty:
                        continue
                    f.add_trace(go.Bar(x=[hname(h) for h in s1["horizon"]], y=(s1[col] * 100.0).tolist(), name=lib.METHOD_NAME[m], marker_color=lib.METHOD_COLOR[m],
                                       customdata=s1["cases"].tolist(), hovertemplate="%{y:.1f}% (%{customdata} cases)<extra></extra>"))
                f.add_trace(go.Scatter(x=[hname(h) for h in hz_all], y=[target] * len(hz_all), mode="lines", line=dict(dash="dash", color="#B5483A"),
                                       name=f"target {target}%", hovertemplate=f"target {target}%<extra></extra>"))
                f.update_layout(title=title, height=340, barmode="group", yaxis=dict(range=[0, 100], title="%"), margin=dict(l=10, r=10, t=40, b=10),
                                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0))
                return f
            a, b = st.columns(2)
            a.plotly_chart(bars("hit50", "Real price inside the 50% range (should be near 50%)", 50), **WIDE, key="fc_bars50")
            b.plotly_chart(bars("hit80", "Real price inside the 80% range (should be near 80%)", 80), **WIDE, key="fc_bars80")

            hz = st.selectbox("Accuracy as it comes in, looking ahead", hz_all, format_func=hname, key="fc_horizon")
            if hz not in hz_all:
                hz = hz_all[0]
            ot = lib.accuracy_over_time(ledger, int(hz))

            def running(col, title, target, label):
                f = go.Figure()
                for m in lib.METHODS:
                    s1 = ot[ot["method"] == m]
                    if s1.empty:
                        continue
                    f.add_trace(go.Scatter(x=pd.to_datetime(s1["target_day"]), y=s1[col], mode="lines+markers", name=lib.METHOD_NAME[m], line=dict(color=lib.METHOD_COLOR[m]),
                                           customdata=s1["cases"].tolist(), hovertemplate=label + " after %{customdata} graded cases<extra></extra>"))
                if len(ot):
                    x = pd.to_datetime(ot["target_day"])
                    f.add_trace(go.Scatter(x=[x.min(), x.max()], y=[target, target], mode="lines", line=dict(dash="dash", color="#B5483A"), name="target", hoverinfo="skip"))
                f.update_layout(title=title, height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0))
                return f
            a, b = st.columns(2)
            a.plotly_chart(running("inside80", f"Share inside the 80% range, running total ({hname(hz)} ahead)", 80, "%{y:.1f}% inside"), **WIDE, key="fc_run80")
            b.plotly_chart(running("vs_naive", f"Closer than staying put? Running ratio, below 1.00 is better ({hname(hz)} ahead)", 1.0, "ratio %{y:.2f}"), **WIDE, key="fc_runnaive")
            st.caption("Each point is the running total over every forecast whose day had arrived by that date, so early points move a lot and settle as cases accumulate.")
            with st.expander("By price level (the item's daily price when the guess was made)"):
                lvt = lib.score_table(board["levels"])
                st.dataframe(lvt, hide_index=True, **WIDE, column_config=score_config(lvt))
                st.caption("Cheap items usually swing the most, so expect wider ranges and larger misses there.")

# ---------------------------------------------------------------- Confidence
with tab_conf:
    mix = lib.confidence_mix(hist)
    st.write("Confidence is earned item by item. An item is High only after your own sales agreed with the tool's prediction for it.")
    a, b = st.columns(2)
    fig = go.Figure()
    for tier in lib.CONF_ORDER:
        m = mix[mix["tier"] == tier]
        fig.add_trace(go.Scatter(x=m["data_last_ts"], y=m["items"], name=tier, stackgroup="one", line=dict(color=lib.CONF_COLOR[tier])))
    fig.update_layout(title="Items by confidence label over time", height=340, margin=dict(l=10, r=10, t=40, b=10))
    a.plotly_chart(fig, **WIDE, key="conf_mix")
    if "median_range_ratio" in hist:
        fig = go.Figure(go.Scatter(x=hist["data_last_ts"], y=hist["median_range_ratio"], line=dict(color="#2B4C59")))
        fig.update_layout(title="Median range width (high / low sales per day). Smaller = tighter error bars", height=340,
                          margin=dict(l=10, r=10, t=40, b=10))
        b.plotly_chart(fig, **WIDE, key="conf_range")
    st.markdown("""
**How labels are decided**
- **Low**: under 24 hours of data; lumpy and slow demand (testing showed these were overstated); a range more than 5x wide; or it sold nothing while first in line although the tool predicted a sale.
- **Medium**: everything else. The highest label without a real-sales check.
- **High**: your experiment checked this item and the predicted time fell inside what happened, with no check pointing the other way.
""")

# ---------------------------------------------------------------- Assumptions (footnote under every tab)
st.divider()
st.caption("**Assumptions behind every profit figure on this page.** " + " ".join(HFL.assumptions()))
