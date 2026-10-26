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
    """Converts a decimal gold float into a formatted g s c string."""
    if pd.isna(gold_float):
        return "0g 0s 0c"
    
    # Convert float back to total copper
    copper_total = int(round(gold_float * 10000))
    
    g = copper_total // 10000
    s = (copper_total % 10000) // 100
    c = copper_total % 100
    
    return f"{g:,}g {s}s {c}c"

@st.cache_data
def load_metadata():
    try:
        temp_db = duckdb.connect(':memory:')
        # Added 'name' so we can populate the dropdown menus natively
        query = """
            SELECT 
                itemId as item_id, 
                name as item_name,
                class as item_class, 
                subclass as item_subclass 
            FROM read_json_auto('items.json')
        """
        df_meta = temp_db.execute(query).df()
        temp_db.close()
        return df_meta
    except Exception as e:
        st.error(f"Could not load items.json. Did you push it to GitHub? Error: {e}")
        return pd.DataFrame(columns=['item_id', 'item_name', 'item_class', 'item_subclass'])

def prep_treemap_data(auction_df, meta_df):
    df = pd.merge(auction_df, meta_df, left_on='Item_ID', right_on='item_id', how='left')
    
    latest_scan = df['timestamp'].max()
    current_df = df[df['timestamp'] == latest_scan]
    historical_df = df[df['timestamp'] < latest_scan]
    
    # Added Item_ID to the groupby so it isn't lost during aggregation
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
        trend_df,
        path=[px.Constant("All Categories"), 'item_class', 'item_subclass', 'Item_Name'],
        values='total_volume',
        color='price_trend_pct',
        # Passed Item_ID into custom_data for the tooltip
        custom_data=['Formatted_Price', 'Item_ID'], 
        color_continuous_scale=['#B33F3F', '#2B4C59', '#3E7C59'], 
        color_continuous_midpoint=0,
        range_color=[-100, 100] 
    )
    
    # Updated Hovertemplate to display the ID
    fig.update_traces(
        hovertemplate="<b>%{label}</b> (ID: %{customdata[1]})<br>Volume: %{value}<br>Price: %{customdata[0]}<br>Trend: %{color:.1f}%<extra></extra>"
    )
    
    fig.update_layout(
        margin=dict(t=30, l=10, r=10, b=10), 
        paper_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False 
    )
    return fig

# ---------------------------------------------------------
# 1. Setup the Web Page
# ---------------------------------------------------------
st.set_page_config(page_title="WoW Market Dashboard", layout="wide")
st.title("World of Warcraft Market Data")
st.markdown("Live auction house data querying directly from Backblaze B2.")

# Load Item Metadata
meta_df = load_metadata()

# ---------------------------------------------------------
# 2. Connect to DuckDB and Backblaze
# ---------------------------------------------------------
@st.cache_resource
def connect_to_datalake():
    con = duckdb.connect(database=':memory:')
    con.execute("INSTALL httpfs; LOAD httpfs;")
    
    b2_key = st.secrets["B2_KEY_ID"]
    b2_secret = st.secrets["B2_APP_KEY"]
    
    con.execute(f"""
        CREATE OR REPLACE SECRET b2_secret (
            TYPE S3,
            KEY_ID '{b2_key}',
            SECRET '{b2_secret}',
            ENDPOINT 's3.us-east-005.backblazeb2.com',
            URL_STYLE 'path'
        );
    """)
    return con

try:
    db = connect_to_datalake()
    st.success("Successfully connected to the Data Lake!")
except Exception as e:
    st.error(f"Failed to connect: {e}")

# ---------------------------------------------------------
# 3. Global UI Controls
# ---------------------------------------------------------
realm_options = {
    "Realm 73": "realm_73",
    "Realm 96": "realm_96",
    "Realm 118": "realm_118",
    "Commodities": "commodities"
}
selected_display_name = st.selectbox("Select Realm/Category", list(realm_options.keys()))
target_folder = realm_options[selected_display_name]

s3_path = f"s3://wow-market-data-lake/{target_folder}/**/*.parquet"
dict_path = "s3://wow-market-data-lake/system/item_dictionary.parquet"
price_col = "unit_price" if target_folder == "commodities" else "buyout"

# ---------------------------------------------------------
# 4. Tabs & Dynamic Query Execution
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["Item Lookup", "Overall Market Trends"])

# ==========================================
# TAB 1: SPECIFIC ITEM LOOKUP
# ==========================================
with tab1:
    if meta_df.empty:
        st.warning("Please ensure items.json is present in the repository to use the item browser.")
    else:
        # Category Drill-Down Columns
        cat_col, sub_col, item_col = st.columns(3)
        
        with cat_col:
            classes = sorted(meta_df['item_class'].dropna().unique().tolist())
            selected_class = st.selectbox("1. Select Category", classes)
            
        with sub_col:
            subclasses = sorted(meta_df[meta_df['item_class'] == selected_class]['item_subclass'].dropna().unique().tolist())
            selected_subclass = st.selectbox("2. Select Subcategory", subclasses)
            
        with item_col:
            # Filter and construct the item dictionary { "Item Name (ID)": ID }
            items_in_subclass = meta_df[(meta_df['item_class'] == selected_class) & (meta_df['item_subclass'] == selected_subclass)]
            items_in_subclass = items_in_subclass.dropna(subset=['item_name']).sort_values('item_name')
            
            # Appending ID to the string prevents dictionary collisions if two items share the same name
            item_display = items_in_subclass.apply(lambda x: f"{x['item_name']} ({x['item_id']})", axis=1)
            item_options = dict(zip(item_display, items_in_subclass['item_id']))
            
            selected_item_display = st.selectbox("3. Select Item", list(item_options.keys()))
    
    if st.button(f"Fetch Item Data for {selected_display_name}"):
        if not selected_item_display:
            st.warning("Please select an item from the dropdown.")
        else:
            # Pull the exact ID based on what was selected in the third column
            search_id = item_options[selected_item_display]
            
            with st.spinner(f"Compiling historical market data for {selected_item_display}..."):
                query = f"""
                    SELECT 
                        COALESCE(dict.name, 'Unknown Item') as Item_Name,
                        auc.item.id as Item_ID, 
                        (auc.{price_col} / 10000.0) as Gold_Price, 
                        auc.quantity as Quantity,
                        COALESCE(auc.scan_timestamp, make_timestamp(year, month, day, 0, 0, 0)) as timestamp
                    FROM read_parquet('{s3_path}', hive_partitioning=1, union_by_name=True) as auc
                    LEFT JOIN read_parquet('{dict_path}') as dict 
                        ON auc.item.id = dict.id
                    WHERE auc.item.id = {search_id}
                    ORDER BY timestamp ASC;
                """
                
                try:
                    df = db.execute(query).df()
