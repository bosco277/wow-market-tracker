import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 1. Setup the Web Page
st.set_page_config(page_title="WoW Market Dashboard", layout="wide")
st.title("World of Warcraft Market Data")
st.markdown("Live auction house data querying directly from Backblaze B2.")

# 2. Connect to DuckDB and Backblaze
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

# 3. Interactive UI Controls
col1, col2 = st.columns(2)

with col1:
    realm_options = {
        "Realm 73": "realm_73",
        "Realm 96": "realm_96",
        "Realm 118": "realm_118",
        "Commodities": "commodities"
    }
    selected_display_name = st.selectbox("Select Realm/Category", list(realm_options.keys()))
    target_folder = realm_options[selected_display_name]

with col2:
    search_id = st.text_input("Search Item ID (Optional)", placeholder="e.g., 208396")

# 4. Dynamic Query Execution & Dashboard
search_term = search_id.strip()

if st.button(f"Fetch Data for {selected_display_name}") and search_term:
    with st.spinner("Compiling historical market data..."):
        
        s3_path = f"s3://wow-market-data-lake/{target_folder}/**/*.parquet"
        dict_path = "s3://wow-market-data-lake/system/item_dictionary.parquet"
        
        # Dynamically handle Blizzard's two different price columns
        price_col = "unit_price" if target_folder == "commodities" else "buyout"
        
        # Fetch Historical Data with hive_partitioning to extract timestamps
        query = f"""
            SELECT 
                COALESCE(dict.name, 'Unknown Item') as Item_Name,
                auc.item.id as Item_ID, 
                (auc.{price_col} / 10000.0) as Gold_Price, 
                auc.quantity as Quantity,
                make_timestamp(year, month, day, hour, 0, 0) as timestamp
            FROM read_parquet('{s3_path}', hive_partitioning=1) as auc
            LEFT JOIN read_parquet('{dict_path}') as dict 
                ON auc.item.id = dict.id
            WHERE dict.name ILIKE '%{search_term}%' OR auc.item.id = TRY_CAST('{search_term}' AS BIGINT)
            ORDER BY timestamp ASC;
        """
