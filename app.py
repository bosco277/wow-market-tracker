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

# 3. Create a Basic Query Button for Realm 73
if st.button("Fetch Latest Auctions for Realm 73"):
    with st.spinner("Querying Parquet files..."):
        query = """
            SELECT item.id, item.context, buyout, quantity 
            FROM read_parquet('s3://wow-market-data-lake/realm_73/**/*.parquet') 
            LIMIT 50;
        """
        # Execute the DuckDB query and convert it instantly to a Pandas DataFrame
        df = db.execute(query).df()
        
        # Display the DataFrame as an interactive web table
        st.dataframe(df, use_container_width=True)
