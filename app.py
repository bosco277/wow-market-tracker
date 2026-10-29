import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import boto3
import io
import duckdb

# ---------------------------------------------------------
# 1. Setup & Configuration (Bloomberg Terminal Styling)
# ---------------------------------------------------------
st.set_page_config(page_title="WoW Market Analytics", layout="wide", initial_sidebar_state="collapsed")
st.title("Algorithmic Trading Suite")

# Helper for Gold formatting
def format_gsc(gold_float):
    if pd.isna(gold_float):
        return "0g"
    return f"{gold_float:,.2f}g"

# ---------------------------------------------------------
# 2. Data Ingestion (Direct from B2)
# ---------------------------------------------------------
@st.cache_data(ttl=600) # Cache for 10 minutes to minimize B2 read operations
def load_dashboard_payloads():
    s3 = boto3.client(
        "s3",
        endpoint_url="https://s3.us-east-005.backblazeb2.com",
        aws_access_key_id=st.secrets["B2_KEY_ID"],
        aws_secret_access_key=st.secrets["B2_APP_KEY"]
    )
    
    def fetch_parquet(key):
        try:
            obj = s3.get_object(Bucket="wow-market-data-lake", Key=key)
            return pd.read_parquet(io.BytesIO(obj['Body'].read()))
        except Exception:
            return pd.DataFrame()

    screener = fetch_parquet("dashboard_screener.parquet")
    charts = fetch_parquet("dashboard_charts.parquet")
    dict_df = fetch_parquet("system/item_dictionary.parquet")
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

with st.spinner("Fetching live market payloads..."):
    screener_df, charts_df, dict_df = load_dashboard_payloads()
    meta_df = load_metadata()

# Merge naming and category dictionaries into the screener payload
if not screener_df.empty:
    if not dict_df.empty:
        screener_df = pd.merge(screener_df, dict_df, left_on='item_id', right_on='id', how='left')
        screener_df.rename(columns={'name': 'item_name'}, inplace=True)
    if not meta_df.empty:
        screener_df = pd.merge(screener_df, meta_df[['item_id', 'item_class', 'item_subclass']], on='item_id', how='left')
        
    # Calculate Arbitrage Metrics
    if 'floor_price' in screener_df.columns:
        screener_df['discount_to_vwap'] = ((screener_df['liquidity_adjusted_vwap'] - screener_df['floor_price']) / screener_df['liquidity_adjusted_vwap']) * 100
        screener_df['discount_to_vwap'] = screener_df['discount_to_vwap'].fillna(0).round(2)
        screener_df['action_signal'] = screener_df.apply(lambda x: 'BUY (DISCOUNT)' if x['floor_price'] <= x['buy_threshold'] else ('OVERPRICED' if x['floor_price'] > x['liquidity_adjusted_vwap'] else 'HOLD'), axis=1)
    else:
        screener_df['discount_to_vwap'] = 0.0
        screener_df['action_signal'] = 'DATA PENDING'
        
    screener_df['item_name'] = screener_df['item_name'].fillna('Unknown Item')

# ---------------------------------------------------------
# 3. Terminal Interface
# ---------------------------------------------------------
tab_screener, tab_charts, tab_portfolio = st.tabs(["Global Screener", "Visual Analytics", "Portfolio Ledger"])

# --- TAB 1: THE GLOBAL SCREENER ---
with tab_screener:
    st.markdown("### Market Heatmap & Arbitrage Screener")
    if screener_df.empty:
        st.warning("Dashboard payload is currently empty. Run the compaction pipeline.")
    else:
        # Category Filter
        col1, col2 = st.columns([1, 3])
        with col1:
            if 'item_class' in screener_df.columns:
                categories = ['All'] + sorted(screener_df['item_class'].dropna().unique().tolist())
                selected_cat = st.selectbox("Filter by Category", categories)
            else:
                selected_cat = 'All'
        
        display_df = screener_df.copy()
        if selected_cat != 'All':
            display_df = display_df[display_df['item_class'] == selected_cat]
            
        # Display the interactive grid
        grid_columns = ['item_id', 'item_name', 'action_signal', 'discount_to_vwap', 'liquidity_adjusted_vwap', 'buy_threshold', 'sell_threshold', 'target_volume']
        if 'floor_price' in display_df.columns:
            grid_columns.insert(4, 'floor_price')
            
        st.dataframe(
            display_df[grid_columns].sort_values('discount_to_vwap', ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "discount_to_vwap": st.column_config.NumberColumn("Discount to VWAP (%)", format="%.2f%%"),
                "liquidity_adjusted_vwap": st.column_config.NumberColumn("LA-VWAP", format="%.2f g"),
                "floor_price": st.column_config.NumberColumn("Current Floor", format="%.2f g"),
                "buy_threshold": st.column_config.NumberColumn("Target Buy", format="%.2f g"),
                "sell_threshold": st.column_config.NumberColumn("Target Sell", format="%.2f g"),
                "action_signal": st.column_config.TextColumn("System Signal")
            }
        )

# --- TAB 2: VISUAL ANALYTICS ---
with tab_charts:
    st.markdown("### Historical Liquidity Validation")
    if not charts_df.empty and not screener_df.empty:
        # Create a dictionary for the dropdown: "Item Name (ID)"
        item_list = screener_df.dropna(subset=['item_name']).sort_values('item_name')
        item_options = dict(zip(item_list['item_name'] + " (" + item_list['item_id'].astype(str) + ")", item_list['item_id']))
        
        selected_chart_item = st.selectbox("Select Asset to Analyze", list(item_options.keys()))
        target_id = item_options[selected_chart_item]
        
        # Filter the 7-day chart payload
        plot_df = charts_df[charts_df['item_id'] == target_id].copy()
        
        if not plot_df.empty:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # The yellow VWAP line
            fig.add_trace(go.Scatter(
                x=plot_df['scan_timestamp'], y=plot_df['liquidity_adjusted_vwap'],
                name='LA-VWAP', mode='lines', line=dict(color='#F2C94C', width=3)
            ), secondary_y=False)
            
            # The raw market floor (if available)
            if 'floor_price' in plot_df.columns:
                fig.add_trace(go.Scatter(
                    x=plot_df['scan_timestamp'], y=plot_df['floor_price'],
                    name='Market Floor', mode='lines+markers', line=dict(color='#2D9CDB', width=1),
                    marker=dict(size=4)
                ), secondary_y=False)
            
            # Volume bars on secondary axis
            if 'target_volume' in plot_df.columns:
                fig.add_trace(go.Bar(
                    x=plot_df['scan_timestamp'], y=plot_df['target_volume'],
                    name='24h Volume', marker_color='rgba(255, 255, 255, 0.1)'
                ), secondary_y=True)

            fig.update_layout(
                title=f"7-Day Liquidity Analysis: {selected_chart_item}",
                plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
                hovermode='x unified', margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.update_yaxes(title_text="Price (Gold)", secondary_y=False, rangemode='tozero')
            fig.update_yaxes(showgrid=False, showticklabels=False, secondary_y=True)
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 7-day historical chart data found for this asset.")

# --- TAB 3: PORTFOLIO LEDGER ---
with tab_portfolio:
    st.markdown("### Active Inventory & Cost Basis Tracking")
    st.caption("Input your current inventory manually. The system will calculate real-time profitability against live algorithms.")
    
    # Initialize session state for the interactive ledger
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = pd.DataFrame(columns=["item_id", "quantity", "avg_cost_basis"])
        
    # Interactive Data Editor
    edited_portfolio = st.data_editor(
        st.session_state["portfolio"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "item_id": st.column_config.NumberColumn("Item ID", required=True),
            "quantity": st.column_config.NumberColumn("Total Quantity", min_value=1, required=True),
            "avg_cost_basis": st.column_config.NumberColumn("Avg Cost per Unit (Gold)", min_value=0.0, format="%.4f", required=True)
        }
    )
    
    # Process portfolio metrics
    if not edited_portfolio.empty and not screener_df.empty:
        st.session_state["portfolio"] = edited_portfolio
        
        # Merge portfolio inputs with live algorithms
        ledger = pd.merge(edited_portfolio, screener_df[['item_id', 'item_name', 'liquidity_adjusted_vwap', 'floor_price']], on='item_id', how='left')
        
        # Financial Math (Accounting for 5% AH Tax and demanding 2% profit)
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
        
        st.markdown("#### Live Portfolio Execution Targets")
        display_ledger = ledger[['item_name', 'quantity', 'avg_cost_basis', 'target_sell_price', 'floor_price', 'unrealized_profit', 'roi_pct', 'execution_signal']]
        st.dataframe(
            display_ledger,
            use_container_width=True,
            hide_index=True,
            column_config={
                "avg_cost_basis": st.column_config.NumberColumn("Your Cost", format="%.2f g"),
                "target_sell_price": st.column_config.NumberColumn("Mathematical Sell Floor", format="%.2f g"),
                "floor_price": st.column_config.NumberColumn("Current Market", format="%.2f g"),
                "unrealized_profit": st.column_config.NumberColumn("Net Profit (After Tax)", format="%.2f g"),
                "roi_pct": st.column_config.NumberColumn("ROI", format="%.2f%%"),
                "execution_signal": st.column_config.TextColumn("Algorithm Signal")
            }
        )
