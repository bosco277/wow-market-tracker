import streamlit as st
import duckdb
import pandas as pd

# 1. Setup the Web Page
st.set_page_config(page_title="WoW Market Dashboard", layout="wide")
st.title("World of Warcraft Market Data")
st.markdown("Live auction house data querying directly from Backblaze B2.")

# 2. Connect to DuckDB and Backblaze
# Streamlit has a special st.cache_resource command so it doesn't reconnect on every button click
@st.cache_resource
def connect_to_datalake():
    con = duckdb.connect(database=':memory:')
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    
    # Streamlit pulls secrets differently than GitHub Actions using st.secrets
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
# Create two columns side-by-side
col1, col2 = st.columns(2)

with col1:
    # A dictionary mapping the dropdown display names to your Backblaze folder names
    realm_options = {
        "Realm 73": "realm_73",
        "Realm 96": "realm_96",
        "Realm 118": "realm_118",
        "Commodities": "commodities"
    }
    selected_display_name = st.selectbox("Select Realm/Category", list(realm_options.keys()))
    target_folder = realm_options[selected_display_name]

with col2:
    # Text input for the Item ID
    search_id = st.text_input("Search Item ID (Optional)", placeholder="e.g., 208396")

# 4. Dynamic Query Execution
if st.button(f"Fetch Data for {selected_display_name}"):
    with st.spinner("Scanning Parquet files in the Data Lake..."):
        
        s3_path = f"s3://wow-market-data-lake/{target_folder}/**/*.parquet"
        dict_path = "s3://wow-market-data-lake/system/item_dictionary.parquet"
        
        # Modify the query based on whether the user typed an Item ID or a Name
        search_term = search_id.strip()
        
        if search_term:
            if search_term.isdigit():
                # It is an Item ID number
                where_clause = f"WHERE auc.item.id = {search_term}"
            else:
                # It is text, so search the dictionary names
                where_clause = f"WHERE dict.name ILIKE '%{search_term}%'"
            limit_clause = "LIMIT 1000"
        else:
            where_clause = ""
            limit_clause = "LIMIT 50"

        # The LEFT JOIN connects the two files instantly
        query = f"""
            SELECT 
                COALESCE(dict.name, 'Unknown Item (' || auc.item.id || ')') as Item_Name,
                auc.item.id as Item_ID, 
                auc.buyout as Buyout_Copper, 
                auc.quantity as Quantity
            FROM read_parquet('{s3_path}') as auc
            LEFT JOIN read_parquet('{dict_path}') as dict 
                ON auc.item.id = dict.id
            {where_clause}
            {limit_clause};
        """
        
        try:
            df = db.execute(query).df()
            
            if df.empty:
                st.warning("No active auctions found for that search.")
            else:
                if 'Buyout_Copper' in df.columns:
                    df['Gold_Price'] = df['Buyout_Copper'] / 10000 
                    
                st.dataframe(df, use_container_width=True)
                st.caption(f"Showing {len(df)} results.")
                
        except Exception as e:
            st.error(f"Query Failed: {e}")
