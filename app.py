import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from huggingface_hub import hf_hub_download
import duckdb
import io

# ---------------------------------------------------------
# 1. Setup & Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="WoW Market Analytics", layout="wide", initial_sidebar_state="collapsed")
st.title("Algorithmic Trading Suite")

# GSC Formatter for precise UI legibility
def format_gsc(gold_float):
    if pd.isna(gold_float) or gold_float == 0:
        return "0g 0s 0c"
    copper_total = int(round(gold_float * 10000))
    g = copper_total // 10000
    s = (copper_total % 10000) // 100
    c = copper_total % 100
    
    parts = []
    if g > 0: parts.append(f"{g:,}g")
    if s > 0 or g > 0: parts.append(f"{s}s")
    parts.append(f"{c}c")
    return " ".join(parts)

# ---------------------------------------------------------
# 2. Data Ingestion (B2 Primary with HF Fallback)
# ---------------------------------------------------------
@st.cache_data(ttl=600)
def load_dashboard_payloads():
    con = duckdb.connect(database=':memory:')
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"""
        CREATE OR REPLACE SECRET b2_secret (
            TYPE S3, KEY_ID '{st.secrets["B2_KEY_ID"]}', SECRET '{st.secrets["B2_APP_KEY"]}',
            ENDPOINT 's3.us-east-005.backblazeb2.com', URL_STYLE 'path'
        );
    """)
    
    def fetch_parquet(filename):
        try:
            query = f"SELECT * FROM read_parquet('s3://wow-market-data-lake/{filename}')"
            return con.execute(query).df()
        except Exception as b2_error:
            try:
                file_path = hf_hub_download(
                    repo_id="bosco277/wow-market-data-lake", 
                    repo_type="dataset", 
                    filename=filename,
                    token=st.secrets.get("HF_TOKEN")
                )
                return pd.read_parquet(file_path)
            except Exception as hf_error:
                if "item_dictionary" in filename:
                    return pd.DataFrame()
                else:
                    st.error(f"Critical Data Outage for {filename}.\nB2 Error: {b2_error}\nHF Error: {hf_error}")
                    return pd.DataFrame()

    screener = fetch_parquet("dashboard_screener.parquet")
    charts = fetch_parquet("dashboard_charts.parquet")
    dict_df = fetch_parquet("system/item_dictionary.parquet")
    
    con.close()
    return screener, charts, dict_df

@st.cache_data
def load_metadata():
    try:
        con = duckdb.connect(':memory:')
        query = "SELECT itemId as item_id, name as item_name, class as item_class, subclass as item_subclass FROM read_json_auto('items.json')"
        df = con.execute(query).df()
        con.close()
        return df
    except Exception:
        return pd.DataFrame(columns=['item_id', 'item_name', 'item_class', 'item_subclass'])

if st.sidebar.button("Force Refresh Data"):
    st.cache_data.clear()

with st.spinner("Fetching live market payloads..."):
    screener_df, charts_df, dict_df = load_dashboard_payloads()
    meta_df = load_metadata()

# ---------------------------------------------------------
# 3. Data Processing & Asset Classification
# ---------------------------------------------------------
if not screener_df.empty:
    if not dict_df.empty and 'id' in dict_df.columns:
        screener_df = pd.merge(screener_df, dict_df, left_on='item_id', right_on='id', how='left')
        screener_df.rename(columns={'name': 'item_name'}, inplace=True)
        
    if not meta_df.empty:
        cols_to_merge = ['item_id', 'item_class', 'item_subclass']
        if 'item_name' not in screener_df.columns:
            cols_to_merge.append('item_name')
        screener_df = pd.merge(screener_df, meta_df[cols_to_merge], on='item_id', how='left')
        
    if 'floor_price' in screener_df.columns and 'liquidity_adjusted_vwap' in screener_df.columns:
        screener_df['discount_to_vwap'] = ((screener_df['liquidity_adjusted_vwap'] - screener_df['floor_price']) / screener_df['liquidity_adjusted_vwap']) * 100
        screener_df['discount_to_vwap'] = screener_df['discount_to_vwap'].fillna(0).round(2)
        screener_df['action_signal'] = screener_df.apply(lambda x: 'BUY (DISCOUNT)' if x['floor_price'] <= x['buy_threshold'] else ('OVERPRICED' if x['floor_price'] > x['liquidity_adjusted_vwap'] else 'HOLD'), axis=1)
    else:
        screener_df['discount_to_vwap'] = 0.0
        screener_df['action_signal'] = 'DATA PENDING'
        
    if 'item_name' not in screener_df.columns:
        screener_df['item_name'] = 'Unknown Item'
    else:
        screener_df['item_name'] = screener_df['item_name'].fillna('Unknown Item')
        
    # Generate Sales Profiles based on velocity and price anomalies
    def classify_asset(row):
        vol = row.get('target_volume', 0)
        price = row.get('liquidity_adjusted_vwap', 0)
        
        if vol < 5 and price > 25000:
            return "Anomaly / Transfer"
        elif vol >= 1000:
            return "Daily Staple"
        elif vol >= 100:
            return "Cyclical Purchase"
        elif vol >= 10:
            return "Slow Mover"
        elif vol > 0:
            return "Illiquid / Rare"
        else:
            return "Dead Market"
            
    screener_df['sales_profile'] = screener_df.apply(classify_asset, axis=1)

# ---------------------------------------------------------
# 4. Terminal Interface
# ---------------------------------------------------------
tab_screener, tab_charts, tab_portfolio = st.tabs(["Global Screener", "Visual Analytics", "Portfolio Ledger"])

# --- TAB 1: THE GLOBAL SCREENER ---
with tab_screener:
    if screener_df.empty:
        st.warning("Dashboard payload is currently empty. Run the compaction pipeline.")
    else:
        st.markdown("### Market Controls")
        
        # Layout for Filters
        col_cat, col_prof, col_price = st.columns([1, 1, 2])
        
        with col_cat:
            categories = ['All'] + sorted(screener_df['item_class'].dropna().astype(str).unique().tolist())
            selected_cat = st.selectbox("Category", categories)
            
        with col_prof:
            profiles = ['All', 'Daily Staple', 'Cyclical Purchase', 'Slow Mover', 'Illiquid / Rare', 'Anomaly / Transfer']
            selected_prof = st.selectbox("Sales Profile", profiles, index=0)
            
        with col_price:
            if 'floor_price' in screener_df.columns:
                min_price = float(screener_df['floor_price'].min())
                max_price = float(screener_df['floor_price'].max())
                # Cap the slider display at a reasonable threshold to prevent anomalies from ruining the scale
                slider_max = min(max_price, 500000.0) 
                price_range = st.slider("Current Floor Bound (Gold)", min_value=0.0, max_value=slider_max, value=(0.0, slider_max))
            else:
                price_range = (0.0, 9999999.0)

        # Apply Filters
        display_df = screener_df.copy()
        if selected_cat != 'All':
            display_df = display_df[display_df['item_class'] == selected_cat]
        if selected_prof != 'All':
            display_df = display_df[display_df['sales_profile'] == selected_prof]
        if 'floor_price' in display_df.columns:
            display_df = display_df[(display_df['floor_price'] >= price_range[0]) & (display_df['floor_price'] <= price_range[1])]

        st.divider()

        # Treemap Integration
        st.markdown("### Asset Allocation Treemap")
        if not display_df.empty and 'item_class' in display_df.columns:
            # Ensure safe data for treemap
            tree_df = display_df.copy()
            tree_df['item_class'] = tree_df['item_class'].fillna('Unknown')
            tree_df['item_subclass'] = tree_df['item_subclass'].fillna('Unknown')
            tree_df['target_volume'] = tree_df['target_volume'].fillna(1).clip(lower=1) # Treemap sizes cannot be 0
            
            fig_tree = px.treemap(
                tree_df, 
                path=[px.Constant("Filtered Market"), 'item_class', 'item_subclass', 'item_name'],
                values='target_volume',
                color='discount_to_vwap',
                color_continuous_scale=['#B33F3F', '#F2EDE4', '#3E7C59'],
                color_continuous_midpoint=0,
                range_color=[-20, 20],
                hover_data={'action_signal': True, 'discount_to_vwap': ':.2f'}
            )
            fig_tree.update_layout(margin=dict(t=10, l=10, r=10, b=10), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_tree, use_container_width=True)

        # Data Grid
        st.markdown("### Filtered Arbitrage Grid")
        grid_columns = ['item_id', 'item_name', 'sales_profile', 'action_signal', 'discount_to_vwap', 'liquidity_adjusted_vwap', 'buy_threshold', 'sell_threshold', 'target_volume']
        if 'floor_price' in display_df.columns:
            grid_columns.insert(5, 'floor_price')
            
        grid_columns = [col for col in grid_columns if col in display_df.columns]
        sort_col = 'discount_to_vwap' if 'discount_to_vwap' in display_df.columns else grid_columns[0]
            
        st.dataframe(
            display_df[grid_columns].sort_values(by=sort_col, ascending=False),
            width='stretch',
            hide_index=True,
            column_config={
                "discount_to_vwap": st.column_config.NumberColumn("Discount (%)", format="%.2f%%"),
                "liquidity_adjusted_vwap": st.column_config.NumberColumn("LA-VWAP", format="%.2f"),
                "floor_price": st.column_config.NumberColumn("Floor Price", format="%.2f"),
                "buy_threshold": st.column_config.NumberColumn("Target Buy", format="%.2f"),
                "sell_threshold": st.column_config.NumberColumn("Target Sell", format="%.2f"),
                "action_signal": st.column_config.TextColumn("Signal"),
                "sales_profile": st.column_config.TextColumn("Profile")
            }
        )

# --- TAB 2: VISUAL ANALYTICS ---
with tab_charts:
    st.markdown("### Historical Liquidity Validation")
    if not charts_df.empty and not display_df.empty: # Link dropdown strictly to the filtered display_df
        # Create dictionary and clean decimal from item_id
        item_list = display_df.dropna(subset=['item_name']).sort_values('item_name')
        item_list['item_id_str'] = item_list['item_id'].astype(int).astype(str)
        item_options = dict(zip(item_list['item_name'] + " (" + item_list['item_id_str'] + ")", item_list['item_id']))
        
        if item_options:
            selected_chart_item = st.selectbox("Select Asset (Driven by Screener Filters)", list(item_options.keys()))
            target_id = item_options[selected_chart_item]
            
            plot_df = charts_df[charts_df['item_id'] == target_id].copy()
            
            if not plot_df.empty:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(go.Scatter(
                    x=plot_df['scan_timestamp'], y=plot_df['liquidity_adjusted_vwap'],
                    name='LA-VWAP', mode='lines', line=dict(color='#F2C94C', width=3)
                ), secondary_y=False)
                
                if 'floor_price' in plot_df.columns:
                    fig.add_trace(go.Scatter(
                        x=plot_df['scan_timestamp'], y=plot_df['floor_price'],
                        name='Market Floor', mode='lines+markers', line=dict(color='#2D9CDB', width=1),
                        marker=dict(size=4)
                    ), secondary_y=False)
                
                if 'target_volume' in plot_df.columns:
                    fig.add_trace(go.Bar(
                        x=plot_df['scan_timestamp'], y=plot_df['target_volume'],
                        name='24h Volume', marker_color='rgba(0, 0, 0, 0.1)'
                    ), secondary_y=True)

                fig.update_layout(
                    title=f"7-Day Liquidity Analysis: {selected_chart_item}",
                    plot_bgcolor='#FFFFFF', # White chart background
                    paper_bgcolor='#FFFFFF',
                    font=dict(color='#000000'), # Black text for legibility
                    hovermode='x unified', margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig.update_yaxes(title_text="Price (Gold)", secondary_y=False, rangemode='tozero', gridcolor='#E0E0E0')
                fig.update_yaxes(showgrid=False, showticklabels=False, secondary_y=True)
                fig.update_xaxes(gridcolor='#E0E0E0')
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No 7-day historical chart data found for this asset.")
        else:
            st.info("Your current Screener filters are too restrictive. No items to display.")

# --- TAB 3: PORTFOLIO LEDGER ---
with tab_portfolio:
    st.markdown("### Active Inventory & Cost Basis Tracking")
    
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = pd.DataFrame(columns=["item_id", "quantity", "avg_cost_basis"])
        
    edited_portfolio = st.data_editor(
        st.session_state["portfolio"],
        num_rows="dynamic",
        width='stretch',
        column_config={
            "item_id": st.column_config.NumberColumn("Item ID", required=True),
            "quantity": st.column_config.NumberColumn("Total Qty", min_value=1, required=True),
            "avg_cost_basis": st.column_config.NumberColumn("Avg Cost per Unit", min_value=0.0, format="%.2f", required=True)
        }
    )
    
    if not edited_portfolio.empty and not screener_df.empty:
        st.session_state["portfolio"] = edited_portfolio
        
        cols_to_pull = ['item_id', 'item_name', 'liquidity_adjusted_vwap']
        if 'floor_price' in screener_df.columns:
            cols_to_pull.append('floor_price')
            
        ledger = pd.merge(edited_portfolio, screener_df[cols_to_pull], on='item_id', how='left')
        
        ledger['minimum_profitable_price'] = ledger['avg_cost_basis'] / 0.95
        ledger['target_sell_price'] = ledger['minimum_profitable_price'] * 1.02
        
        if 'floor_price' in ledger.columns:
            ledger['current_revenue'] = (ledger['quantity'] * ledger['floor_price']) * 0.95
            ledger['total_cost'] = ledger['quantity'] * ledger['avg_cost_basis']
            ledger['unrealized_profit'] = ledger['current_revenue'] - ledger['total_cost']
            ledger['roi_pct'] = (ledger['unrealized_profit'] / ledger['total_cost']) * 100
            
            ledger['execution_signal'] = ledger.apply(
                lambda x: 'EXECUTE SELL' if (x['floor_price'] >= (x['liquidity_adjusted_vwap'] * 0.98)) and (x['floor_price'] >= x['target_sell_price']) else 'HOLD', 
                axis=1
            )
            
            # Apply GSC Formatting for UI Display
            display_ledger = ledger[['item_name', 'quantity', 'avg_cost_basis', 'target_sell_price', 'floor_price', 'unrealized_profit', 'roi_pct', 'execution_signal']].copy()
            display_ledger['avg_cost_basis'] = display_ledger['avg_cost_basis'].apply(format_gsc)
            display_ledger['target_sell_price'] = display_ledger['target_sell_price'].apply(format_gsc)
            display_ledger['floor_price'] = display_ledger['floor_price'].apply(format_gsc)
            display_ledger['unrealized_profit'] = display_ledger['unrealized_profit'].apply(format_gsc)
            
            st.markdown("#### Live Portfolio Execution Targets")
            st.dataframe(
                display_ledger,
                width='stretch',
                hide_index=True,
                column_config={
                    "avg_cost_basis": st.column_config.TextColumn("Your Cost"),
                    "target_sell_price": st.column_config.TextColumn("Math Sell Floor"),
                    "floor_price": st.column_config.TextColumn("Current Market"),
                    "unrealized_profit": st.column_config.TextColumn("Net Profit (After Tax)"),
                    "roi_pct": st.column_config.NumberColumn("ROI", format="%.2f%%"),
                    "execution_signal": st.column_config.TextColumn("Algorithm Signal")
                }
            )
