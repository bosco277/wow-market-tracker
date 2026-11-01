"""WoW market desk. Reads the small tables the engine writes to engine/ in the Hugging Face dataset repo (not Backblaze).

Every estimate is shown as low - best - high with a confidence label. Nothing here is advice to buy; the ranges are the point.
Click a row in the Market or Flips table to see that item's charts below it."""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lib

HF_REPO = "bosco277/wow-market-data-lake"
st.set_page_config(page_title="WoW Market Desk", layout="wide")


def _stretch() -> dict:
    """Streamlit 1.50+ wants width="stretch"; older versions only know use_container_width. Use whichever this server has."""
    try:
        v = tuple(int(x) for x in st.__version__.split(".")[:2])
    except Exception:
        v = (0, 0)
    return {"width": "stretch"} if v >= (1, 50) else {"use_container_width": True}


WIDE = _stretch()


@st.cache_data(ttl=600, show_spinner="Loading tables...")
def load(name: str) -> pd.DataFrame:
    local = os.environ.get("DERIVED_DIR")
    try:
        if local:
            return pd.read_parquet(os.path.join(local, f"{name}.parquet"))
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HF_REPO, f"engine/{name}.parquet", repo_type="dataset", token=st.secrets.get("HF_TOKEN"))
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


items, hourly, prof = load("items"), load("item_hourly"), load("profile")
cal, hist, exp, flips = load("calibration"), load("calibration_history"), load("experiment"), load("flips")
shelves = load("shelves")
shelves_old = len(shelves) > 0 and "test_regime" not in shelves.columns
shelves_ver = str(shelves["engine_version"].iloc[0]) if len(shelves) and "engine_version" in shelves.columns else "unknown"
shelves = lib.ensure_shelf_columns(shelves)
backtest = load("backtest_summary")
bt_rates = lib.backtest_rates(backtest)
if hist.empty:
    hist = cal

st.title("WoW Market Desk")
hrs, state = lib.freshness(cal)
if items.empty:
    st.error("No tables found yet. Run the 'Market tools' workflow with mode=derive and upload ticked.")
    st.stop()
ver = items["engine_version"].iloc[0] if "engine_version" in items else "?"
st.caption(f"Data {state}: built from snapshots up to {pd.Timestamp(cal['data_last_ts'].max()):%a %b %d %H:%M} UTC "
           f"({hrs:.1f} h ago) | engine {ver} | ranges are low - best - high; labels say how much to trust them")


def column_config(df: pd.DataFrame) -> dict:
    """Numbers stay numbers (so the column sorts properly) and are only formatted for display. Gold is exact to 2 decimals (one silver)."""
    cfg = {}
    for c in df.columns:
        if c == "_id" or not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        if c == "Item ID":
            cfg[c] = st.column_config.NumberColumn(c, format="%d")
        elif c.endswith("(g)"):
            cfg[c] = st.column_config.NumberColumn(c, format="%.2f")
        elif c.endswith("(%)"):
            cfg[c] = st.column_config.NumberColumn(c, format="%.1f%%")
        elif c.endswith(("(units)", "passed", "failed", "warned")):
            cfg[c] = st.column_config.NumberColumn(c, format="%d")
        else:
            cfg[c] = st.column_config.NumberColumn(c, format="%.1f")
    if "Watched" in df.columns:
        cfg["Watched"] = st.column_config.CheckboxColumn("Watched")
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
    fig = go.Figure()
    # Constant lines are drawn as traces (not hline / hrect) so hovering anywhere shows their numbers, same for every point.
    g = lambda v: f"{v:,.2f}g"
    fig.add_trace(go.Scatter(x=h["ts"], y=[r["band_p90"]] * len(h), line=dict(width=0), name="typical high (p90)",
                             hovertemplate="%{y:,.2f}g", showlegend=False))
    fig.add_trace(go.Scatter(x=h["ts"], y=[r["band_p10"]] * len(h), line=dict(width=0), fill="tonexty", fillcolor="rgba(43,76,89,0.12)",
                             name=f"typical band (p10 {g(r['band_p10'])} - p90 {g(r['band_p90'])})", hovertemplate="%{y:,.2f}g"))
    fig.add_trace(go.Scatter(x=h["ts"], y=h["floor_gold"], name="lowest price", line=dict(color="#2B4C59"), hovertemplate="%{y:,.2f}g"))
    sp = h.loc[lib.flag_mask(h, "spike")]
    fig.add_trace(go.Scatter(x=sp["ts"], y=sp["floor_gold"], mode="markers", name="price spike", marker=dict(color="#B5483A", size=8),
                             hovertemplate="%{y:,.2f}g"))
    bl = h.loc[lib.flag_mask(h, "blip")]
    fig.add_trace(go.Scatter(x=bl["ts"], y=bl["floor_gold"], mode="markers", name="listing blip (likely API)",
                             marker=dict(color="#C28A1B", symbol="x", size=8), hovertemplate="%{y:,.2f}g"))
    for col, nm, first in (("wall_low_gold", "wall low", True), ("wall_high_gold", "wall high", False)):
        if col in r and r[col] == r[col]:
            fig.add_trace(go.Scatter(x=h["ts"], y=[r[col]] * len(h), name="price wall (low / high)", legendgroup="wall", showlegend=first,
                                     line=dict(dash="dot", color="#8A9BA3"), hovertemplate=nm + " %{y:,.2f}g"))
    fig.update_layout(title=f"Lowest price per hour (UTC), {len(h)} hours of history", height=420, hovermode="x unified",
                      margin=dict(l=10, r=10, t=40, b=10), yaxis_title="gold",
                      legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0))
    st.plotly_chart(fig, **WIDE, key=f"{key}_price")

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

tab_market, tab_flip, tab_exp, tab_conf = st.tabs(["Market", "Flips", "Experiment", "Confidence"])

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

# ---------------------------------------------------------------- Experiment
with tab_exp:
    sc = lib.experiment_scorecard(cal)
    if exp.empty:
        st.info("No experiment results in the tables yet.")
    else:
        st.write("Your real buy-and-list results checked against what our tool predicted. This is what moves items from Medium to High.")
        if sc.get("rate_n"):
            a, b, c = st.columns(3)
            a.metric("Predicted time inside what happened", f"{sc['inside']} of {sc['rate_n']}")
            b.metric("We predicted too fast", sc["too_fast"])
            c.metric("We predicted too slow", sc["too_slow"])
        if sc.get("unsold_n"):
            st.warning(f"{sc['stayed']} of {sc['unsold_n']} unsold listings sat first in line and still did not sell, although the tool predicted a sale.")
        cols = [c for c in ("posted_utc", "item_id", "lane", "price_gold", "quantity", "units_sold", "units_expired", "appeared_share",
                            "front_share", "still_there_h", "observed_hours_to_clear", "pred_lifo_h_est", "status") if c in exp.columns]
        st.dataframe(exp[cols], hide_index=True, **WIDE)

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
