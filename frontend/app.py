"""WoW market desk. Reads the small tables the engine writes to engine/ in the Hugging Face dataset repo (not Backblaze).

Every estimate is shown as low - best - high with a confidence label. Nothing here is advice to buy; the ranges are the point."""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lib

HF_REPO = "bosco277/wow-market-data-lake"
st.set_page_config(page_title="WoW Market Desk", layout="wide")


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

tab_market, tab_item, tab_exp, tab_conf = st.tabs(["Market", "Item", "Experiment", "Confidence"])

# ---------------------------------------------------------------- Market
with tab_market:
    c1, c2, c3, c4 = st.columns([2, 1, 2, 1])
    q = c1.text_input("Find an item (name or id)")
    minf = c2.number_input("Min price (gold)", 0.0, 100000.0, 0.0, 0.5)
    conf = c3.multiselect("Confidence", lib.CONF_ORDER, default=lib.CONF_ORDER)
    watched = c4.checkbox("Watched only")
    t = lib.market_table(items, minf, tuple(conf), q, watched)
    st.write(f"{len(t):,} items, ranked by estimated gold flow per day (best estimate). The range columns are the honest part.")
    if len(t):
        show = pd.DataFrame({
            "Item": t["name"].where(t["name"] != "", t["item_id"].astype(str)),
            "Lowest price": t["floor_gold"].map(lib.gsc),
            "Vs typical": t.apply(lib.price_position, axis=1),
            "Sold / day (low - best - high)": [lib.rng(a, b, c) for a, b, c in zip(t["daily_low"], t["daily_est"], t["daily_high"])],
            "Gold / day (low - best - high)": [lib.rng(a, b, c) for a, b, c in zip(t["gold_flow_low"], t["gold_flow"], t["gold_flow_high"])],
            "Price wall (est)": t["wall_est_gold"].map(lib.gsc),
            "Confidence": t["confidence"],
            "Why": t["confidence_reason"],
        })
        st.dataframe(show, use_container_width=True, hide_index=True, height=560)

# ---------------------------------------------------------------- Item
with tab_item:
    t = lib.market_table(items)
    labels = {int(r.item_id): (r["name"] if r["name"] else str(r.item_id)) for _, r in t.iterrows()}
    pick = st.selectbox("Item", list(labels), format_func=lambda i: f"{labels[i]} ({i})")
    r = items[items["item_id"] == pick].iloc[0]
    k = st.columns(4)
    k[0].metric("Lowest price now", lib.gsc(r["floor_gold"]), lib.price_position(r))
    k[1].metric("Typical band (p10 - p90)", f"{lib.gsc(r['band_p10'])} - {lib.gsc(r['band_p90'])}")
    k[2].metric("Sold per day (low - best - high)", lib.rng(r["daily_low"], r["daily_est"], r["daily_high"]))
    k[3].markdown(f"**Confidence: <span style='color:{lib.CONF_COLOR[r['confidence']]}'>{r['confidence']}</span>**<br>{r['confidence_reason']}",
                  unsafe_allow_html=True)
    h = hourly[hourly["item_id"] == pick].sort_values("ts")
    fig = go.Figure()
    fig.add_hrect(y0=r["band_p10"], y1=r["band_p90"], fillcolor="#2B4C59", opacity=0.10, line_width=0, annotation_text="typical band")
    fig.add_trace(go.Scatter(x=h["ts"], y=h["floor_gold"], name="lowest price", line=dict(color="#2B4C59")))
    sp = h[h["spike"]]
    fig.add_trace(go.Scatter(x=sp["ts"], y=sp["floor_gold"], mode="markers", name="price spike", marker=dict(color="#B5483A", size=8)))
    bl = h[h["blip"]]
    fig.add_trace(go.Scatter(x=bl["ts"], y=bl["floor_gold"], mode="markers", name="listing blip (likely API)", marker=dict(color="#C28A1B", symbol="x", size=8)))
    for col, nm in (("wall_low_gold", "wall low"), ("wall_high_gold", "wall high")):
        fig.add_hline(y=r[col], line_dash="dot", line_color="#8A9BA3", annotation_text=nm)
    fig.update_layout(title="Lowest price per hour (UTC)", height=380, margin=dict(l=10, r=10, t=40, b=10), yaxis_title="gold")
    st.plotly_chart(fig, use_container_width=True)

    rb = lib.daily_ribbon(h)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=rb["ts"], y=rb["high"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=rb["ts"], y=rb["low"], fill="tonexty", fillcolor="rgba(43,76,89,0.18)", line=dict(width=0), name="low - high"))
    fig2.add_trace(go.Scatter(x=rb["ts"], y=rb["est"], name="best estimate", line=dict(color="#2B4C59")))
    fig2.update_layout(title="Units sold in the last 24 hours (rolling), as a range", height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(f"{len(lib.hole_hours(h))} hours have no estimate (a hole in our data or a flagged interval). The shaded band is the range; "
               "it is not a statistical confidence interval, it is the spread between the cautious and the generous reading of the same data.")

    p = prof[prof["item_id"] == pick] if len(prof) else prof
    if len(p):
        pc = [c for c in ("price_vs_level", "sold_share") if c in p.columns]
        if pc:
            fig3 = go.Figure()
            if "price_vs_level" in p:
                fig3.add_trace(go.Bar(x=p["block"], y=p["price_vs_level"] * 100, name="price vs its own level (%)", marker_color="#2B4C59"))
            if "sold_share" in p:
                fig3.add_trace(go.Bar(x=p["block"], y=p["sold_share"] * 100, name="share of day's sales (%)", marker_color="#C28A1B"))
            fig3.update_layout(barmode="group", title="By hour of day (US Eastern). A draft until a full week of data exists.", height=300,
                               margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig3, use_container_width=True)
    if len(flips):
        f = flips[flips["item_id"] == pick]
        if len(f):
            st.subheader("Past flip-like events")
            st.dataframe(f[[c for c in ("t1", "floor_before_gold", "floor_after_gold", "jump", "units_removed") if c in f.columns]], hide_index=True)

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
        st.dataframe(exp[cols], hide_index=True, use_container_width=True)

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
    a.plotly_chart(fig, use_container_width=True)
    if "median_range_ratio" in hist:
        fig = go.Figure(go.Scatter(x=hist["data_last_ts"], y=hist["median_range_ratio"], line=dict(color="#2B4C59")))
        fig.update_layout(title="Median range width (high / low sales per day). Smaller = tighter error bars", height=340,
                          margin=dict(l=10, r=10, t=40, b=10))
        b.plotly_chart(fig, use_container_width=True)
    st.markdown("""
**How labels are decided**
- **Low**: under 24 hours of data; lumpy and slow demand (testing showed these were overstated); a range more than 5x wide; or it sold nothing while first in line although the tool predicted a sale.
- **Medium**: everything else. The highest label without a real-sales check.
- **High**: your experiment checked this item and the predicted time fell inside what happened, with no check pointing the other way.
""")
