import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json

# ---------------------------------------------------------
# HELPER FUNCTIONS 
# ---------------------------------------------------------
def format_gsc(gold_float):
    if pd.isna(gold_float):
        return "0g 0s 0c"
    copper_total = int(round(gold_float * 10000))
    g = copper_total // 10000
    s = (copper_total % 10000) // 100
    c = copper_total % 100
    return f"{g:,}g {s}s {c}c"

@st.cache_data
def load_metadata():
    try:
        temp_db = duckdb.connect(':memory:')
        query = "SELECT itemId as item_id, name as item_name, class as item_class, subclass as item_subclass FROM read_json_auto('items.json')"
        df_meta = temp_db.execute(query).df()
        temp_db.close()
        return df_meta
    except Exception as e:
        st.error(f"Could not load items.json. Error: {e}")
        return pd.DataFrame(columns=['item_id', 'item_name', 'item_class', 'item_subclass'])

def prep_treemap_data(auction_df, meta_df):
    df = pd.merge(auction_df, meta_df, left_on='Item_ID', right_on='item_id', how='left')
    latest_scan = df['timestamp'].max()
    current_df = df[df['timestamp'] == latest_scan]
    historical_df = df[df['timestamp'] < latest_scan]
    
    current_stats = current_df.groupby(['Item_Name', 'Item_ID', 'item_class', 'item_subclass']).agg(
        current_price=('Gold_Price', 'min'),
        total_volume=('Quantity', 'sum') 
    ).reset_index()
    
    if not historical_df.empty:
        hist_scan_mins = historical_df.groupby(['Item_Name', 'timestamp']).agg(scan_min=('Gold_Price', 'min')).reset_index()
        historical_stats = hist_scan_mins.groupby('Item_Name').agg(hist_price=('scan_min', 'mean')).reset_index()
    else:
        historical_stats = current_stats[['Item_Name', 'current_price']].copy()
        historical_stats.rename(columns={'current_price': 'hist_price'}, inplace=True)
        
    trend_df = pd.merge(current_stats, historical_stats, on='Item_Name')
    trend_df['hist_price'] = trend_df['hist_price'].replace(0, np.nan)
    trend_df['price_trend_pct'] = ((trend_df['current_price'] - trend_df['hist_price']) / trend_df['hist_price']) * 100
    trend_df['price_trend_pct'] = trend_df['price_trend_pct'].fillna(0)
    trend_df.fillna({'item_class': 'Other', 'item_subclass': 'Other', 'Item_ID': 'Unknown'}, inplace=True)
    return trend_df

def create_market_treemap(trend_df):
    trend_df['Formatted_Price'] = trend_df['current_price'].apply(format_gsc)
    fig = px.treemap(
        trend_df, path=[px.Constant("All Categories"), 'item_class', 'item_subclass', 'Item_Name'],
        values='total_volume', color='price_trend_pct', custom_data=['Formatted_Price', 'Item_ID'], 
        color_continuous_scale=['#B33F3F', '#2B4C59', '#3E7C59'], color_continuous_midpoint=0, range_color=[-100, 100] 
    )
    fig.update_traces(hovertemplate="<b>%{label}</b> (ID: %{customdata[1]})<br>Volume: %{value}<br>Price: %{customdata[0]}<br>Trend: %{color:.1f}%<extra></extra>")
    fig.update_layout(margin=dict(t=30, l=10, r=10, b=10), paper_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False)
    return fig

# ---------------------------------------------------------
# 1. Setup the Web Page
# ---------------------------------------------------------
st.set_page_config(page_title="WoW Market Dashboard", layout="wide")
st.title("World of Warcraft Market Data")
st.markdown("Live auction house data querying directly from Backblaze B2.")
meta_df = load_metadata()

# ---------------------------------------------------------
# 2. Connect to DuckDB and Backblaze
# ---------------------------------------------------------
@st.cache_resource
def connect_to_datalake():
    con = duckdb.connect(database=':memory:')
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"""
        CREATE OR REPLACE SECRET b2_secret (
            TYPE S3, KEY_ID '{st.secrets["B2_KEY_ID"]}', SECRET '{st.secrets["B2_APP_KEY"]}',
            ENDPOINT 's3.us-east-005.backblazeb2.com', URL_STYLE 'path'
        );
    """)
    return con

try:
    db = connect_to_datalake()
    st.success("Successfully connected to the Data Lake!")
except Exception as e:
    st.error(f"Failed to connect: {e}")

# ---------------------------------------------------------
# 3. Global UI Controls & Dynamic Schema Check
# ---------------------------------------------------------
realm_options = {"Realm 73": "realm_73", "Realm 96": "realm_96", "Realm 118": "realm_118", "Commodities": "commodities"}
selected_display_name = st.selectbox("Select Realm/Category", list(realm_options.keys()))
target_folder = realm_options[selected_display_name]

s3_path = f"s3://wow-market-data-lake/{target_folder}/year=*/**/*.parquet"
dict_path = "s3://wow-market-data-lake/system/item_dictionary.parquet"
tsm_path = f"s3://wow-market-data-lake/{target_folder}/system/tsm_master.parquet"
price_col = "unit_price" if target_folder == "commodities" else "buyout"

# --- DYNAMIC SCHEMA VALIDATOR ---
with st.spinner("Inspecting database schema for legacy data..."):
    try:
        # Check if the daily files have the new column, the old column, or both
        auc_schema = db.execute(f"DESCRIBE SELECT * FROM read_parquet('{s3_path}', hive_partitioning=1, union_by_name=True)").df()
        auc_cols = auc_schema['column_name'].tolist()
        
        if 'clean_item_id' in auc_cols and 'item' in auc_cols:
            auc_id_sql = "COALESCE(auc.clean_item_id, auc.item.id)"
        elif 'clean_item_id' in auc_cols:
            auc_id_sql = "auc.clean_item_id"
        else:
            auc_id_sql = "auc.item.id"
            
        # Check exactly what the TSM builder named its ID column
        tsm_schema = db.execute(f"DESCRIBE SELECT * FROM read_parquet('{tsm_path}')").df()
        tsm_cols = tsm_schema['column_name'].tolist()
        tsm_id_sql = "tsm.clean_item_id" if "clean_item_id" in tsm_cols else "tsm.Item_ID"
        
    except Exception as e:
        auc_id_sql = "auc.item.id"
        tsm_id_sql = "tsm.Item_ID"

# ---------------------------------------------------------
# 4. Tabs & Dynamic Query Execution
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["Item Lookup", "Overall Market Trends"])

with tab1:
    search_mode = st.radio("Search Mode", ["Text Search (Name or ID)", "Category Browser"], horizontal=True)
    execute_fetch = False
    search_term = None

    if search_mode == "Text Search (Name or ID)":
        col_search, col_btn = st.columns([4, 1])
        with col_search:
            manual_input = st.text_input("Enter Item Name or ID", placeholder="e.g., Linen Cloth or 2589", label_visibility="collapsed")
        with col_btn:
            if st.button("Fetch Item Data", use_container_width=True):
                if manual_input.strip():
                    search_term = manual_input.strip()
                    execute_fetch = True
                else:
                    st.warning("Please enter a valid Item Name or ID.")
    else:
        if meta_df.empty:
            st.warning("Please ensure items.json is present in the repository.")
        else:
            cat_col, sub_col, item_col = st.columns(3)
            with cat_col:
                classes = sorted(meta_df['item_class'].dropna().unique().tolist())
                selected_class = st.selectbox("1. Select Category", classes)
            with sub_col:
                subclasses = sorted(meta_df[meta_df['item_class'] == selected_class]['item_subclass'].dropna().unique().tolist())
                selected_subclass = st.selectbox("2. Select Subcategory", subclasses)
            with item_col:
                items_in_subclass = meta_df[(meta_df['item_class'] == selected_class) & (meta_df['item_subclass'] == selected_subclass)]
                items_in_subclass = items_in_subclass.dropna(subset=['item_name']).sort_values('item_name')
                item_display = items_in_subclass.apply(lambda x: f"{x['item_name']} ({x['item_id']})", axis=1)
                item_options = dict(zip(item_display, items_in_subclass['item_id']))
                selected_item_display = st.selectbox("3. Select Item", list(item_options.keys()))
            if st.button("Fetch Selected Item Data"):
                if selected_item_display:
                    search_term = str(item_options[selected_item_display])
                    execute_fetch = True

    if execute_fetch and search_term:
        with st.spinner(f"Compiling historical market data for {search_term}..."):
            safe_search = search_term.replace("'", "''")
            
            # The bulletproof SQL string formatting
            query = f"""
                SELECT 
                    COALESCE(dict.name, 'Unknown Item') as Item_Name,
                    {auc_id_sql} as Item_ID, 
                    (auc.{price_col} / 10000.0) as Gold_Price, 
                    auc.quantity as Quantity,
                    COALESCE(auc.scan_timestamp, make_timestamp(year, month, day, 0, 0, 0)) as timestamp,
                    tsm.tsm_market_value
                FROM read_parquet('{s3_path}', hive_partitioning=1, union_by_name=True) as auc
                LEFT JOIN read_parquet('{dict_path}') as dict 
                    ON {auc_id_sql} = dict.id
                LEFT JOIN read_parquet('{tsm_path}') as tsm
                    ON {auc_id_sql} = {tsm_id_sql} 
                    AND CAST(COALESCE(auc.scan_timestamp, make_timestamp(year, month, day, 0, 0, 0)) AS DATE) = tsm.date
                WHERE dict.name ILIKE '%{safe_search}%' OR {auc_id_sql} = TRY_CAST('{safe_search}' AS BIGINT)
                ORDER BY timestamp ASC;
            """
            
            try:
                df = db.execute(query).df()
                if not df.empty:
                    if len(df) > 4:
                        median_val = df['Gold_Price'].median()
                        q1 = df['Gold_Price'].quantile(0.25)
                        q3 = df['Gold_Price'].quantile(0.75)
                        iqr = q3 - q1
                        upper_bound = max(q3 + 3 * iqr, median_val * 3)
                        df = df[df['Gold_Price'] <= upper_bound].reset_index(drop=True)
                        
                if not df.empty:
                    df['Hour'] = df['timestamp'].dt.hour
                    df['Day_of_Week'] = df['timestamp'].dt.day_name()
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    
                    st.divider()
                    header_col1, header_col2, header_col3, header_col4 = st.columns([2, 1, 1, 1])
                    with header_col1:
                        st.markdown(f"### {df['Item_Name'].iloc[0]}")
                        st.caption(f"Item ID: {df['Item_ID'].iloc[0]}")
                    with header_col2:
                        current_price = df['Gold_Price'].iloc[-1]
                        st.metric("Market Value", format_gsc(current_price))
                    with header_col3:
                        start_price = df['Gold_Price'].iloc[0]
                        trend = ((current_price - start_price) / start_price) * 100 if start_price else 0
                        st.metric("Trend", f"{trend:+.2f}%")
                    with header_col4:
                        current_qty = df['Quantity'].iloc[-1]
                        st.metric("Available Quantity", f"{current_qty:,}")
                    
                    st.divider()
                    main_col, side_col = st.columns([3, 1])
                    
                    with main_col:
                        st.subheader("Price & Quantity History")
                        agg_df = df.groupby('timestamp').agg(
                            Min_Price=('Gold_Price', 'min'), Total_Quantity=('Quantity', 'sum'),
                            tsm_market_value=('tsm_market_value', 'first')
                        ).reset_index()

                        agg_df['Formatted_Price'] = agg_df['Min_Price'].apply(format_gsc)
                        agg_df['Formatted_TSM'] = agg_df['tsm_market_value'].apply(format_gsc)
                        fig_price = make_subplots(specs=[[{"secondary_y": True}]])
                        fig_price.add_trace(go.Bar(x=agg_df['timestamp'], y=agg_df['Total_Quantity'], name='Total Quantity', marker_color='#C98A4B', opacity=0.4), secondary_y=True)
                        fig_price.add_trace(go.Scatter(x=agg_df['timestamp'], y=agg_df['Min_Price'], name='Min Price', mode='lines', line=dict(color='#2B4C59', width=2), customdata=agg_df['Formatted_Price'], hovertemplate='%{x}<br>Min Price: %{customdata}<extra></extra>'), secondary_y=False)

                        if len(agg_df) > 1:
                            x_numeric = pd.to_numeric(agg_df['timestamp'])
                            z = np.polyfit(x_numeric, agg_df['Min_Price'], 1)
                            trend_vals = np.poly1d(z)(x_numeric)
                            fig_price.add_trace(go.Scatter(x=agg_df['timestamp'], y=trend_vals, name='Linear Price Trend', mode='lines', line=dict(color='#2B4C59', width=2, dash='dot'), customdata=[format_gsc(val) for val in trend_vals], hovertemplate='Trend Price: %{customdata}<extra></extra>'), secondary_y=False)
                            
                        if agg_df['tsm_market_value'].notna().any():
                            fig_price.add_trace(go.Scatter(x=agg_df['timestamp'], y=agg_df['tsm_market_value'], name='TSM Market Value', mode='lines', line=dict(color='#B33F3F', width=3), customdata=agg_df['Formatted_TSM'], hovertemplate='%{x}<br>TSM Value: %{customdata}<extra></extra>'), secondary_y=False)

                        if not agg_df.empty:
                            min_qty = agg_df['Total_Quantity'].min()
                            max_qty = agg_df['Total_Quantity'].max()
                            qty_floor = min_qty * 0.90
                            qty_ceiling = qty_floor + ((max_qty - qty_floor) * 4)

                        fig_price.update_yaxes(title_text="Gold", secondary_y=False, rangemode='tozero')
                        fig_price.update_yaxes(showgrid=False, secondary_y=True, range=[qty_floor, qty_ceiling], showticklabels=False)
                        fig_price.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.plotly_chart(fig_price, width='stretch')
                        
                        st.subheader("Price Heatmap")
                        heatmap_df = df.groupby(['Day_of_Week', 'Hour'])['Gold_Price'].mean().reset_index()
                        heatmap_df['Formatted_Price'] = heatmap_df['Gold_Price'].apply(format_gsc)
                        fig_heat = go.Figure(data=go.Heatmap(z=heatmap_df['Gold_Price'], x=heatmap_df['Hour'], y=heatmap_df['Day_of_Week'], text=heatmap_df['Formatted_Price'], hovertemplate='%{x}:00 %{y}<br>Avg Price: %{text}<extra></extra>', colorscale=[[0, '#F2EDE4'], [1, '#2B4C59']]))
                        fig_heat.update_yaxes(categoryorder='array', categoryarray=day_order[::-1])
                        fig_heat.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0))
                        st.plotly_chart(fig_heat, width='stretch')

                    with side_col:
                        st.subheader("Auction Stats")
                        st.markdown(f"**Min Buyout:** {format_gsc(df['Gold_Price'].min())}  \n**Max Buyout:** {format_gsc(df['Gold_Price'].max())}  \n**Average Price:** {format_gsc(df['Gold_Price'].mean())}  \n**Total Scans:** {len(df):,}  ")
                else:
                    st.warning("No historical data found for that item after filtering.")
            except Exception as e:
                st.error(f"Query Failed: {e}")

with tab2:
    st.subheader(f"Market Trends for {selected_display_name}")
    st.write("Click on a category block to drill down into specific subcategories and items. Color indicates price trend, while block size indicates available quantity.")
    
    if st.button(f"Generate Heatmap for {selected_display_name}"):
        with st.spinner("Aggregating overall market data. This may take a moment..."):
            macro_query = f"""
                SELECT 
                    COALESCE(dict.name, 'Unknown Item') as Item_Name,
                    {auc_id_sql} as Item_ID, 
                    (auc.{price_col} / 10000.0) as Gold_Price, 
                    auc.quantity as Quantity,
                    COALESCE(auc.scan_timestamp, make_timestamp(year, month, day, 0, 0, 0)) as timestamp
                FROM read_parquet('{s3_path}', hive_partitioning=1, union_by_name=True) as auc
                LEFT JOIN read_parquet('{dict_path}') as dict 
                    ON {auc_id_sql} = dict.id
            """
            try:
                macro_df = db.execute(macro_query).df()
                if not macro_df.empty:
                    trend_data = prep_treemap_data(macro_df, meta_df)
                    treemap_fig = create_market_treemap(trend_data)
                    st.plotly_chart(treemap_fig, width='stretch')
                else:
                    st.warning("No data found for this realm/category.")
            except Exception as e:
                st.error(f"Macro Query Failed: {e}")
