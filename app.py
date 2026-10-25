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
    search_id = st.text_input("Search Item ID or Name", placeholder="e.g., Linen Cloth or 2589")

# 4. Dynamic Query Execution & Dashboard
search_term = search_id.strip()

if st.button(f"Fetch Data for {selected_display_name}"):
    
    # Gracefully handle the empty state
    if not search_term:
        st.warning("Please enter an Item ID or Item Name in the search box to view its historical dashboard.")
    else:
        with st.spinner(f"Compiling historical market data for {search_term}..."):
            
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
                    COALESCE(auc.scan_timestamp, make_timestamp(year, month, day, 0, 0, 0)) as timestamp
                FROM read_parquet('{s3_path}', hive_partitioning=1, union_by_name=True) as auc
                LEFT JOIN read_parquet('{dict_path}') as dict 
                    ON auc.item.id = dict.id
                WHERE dict.name ILIKE '%{search_term}%' OR auc.item.id = TRY_CAST('{search_term}' AS BIGINT)
                ORDER BY timestamp ASC;
            """
            
            try:
                df = db.execute(query).df()
                
                if not df.empty:
                    # Extract DateTime features for Heatmaps
                    df['Hour'] = df['timestamp'].dt.hour
                    df['Day_of_Week'] = df['timestamp'].dt.day_name()
                    
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    
                    # Top KPI Header
                    st.divider()
                    header_col1, header_col2, header_col3, header_col4 = st.columns([2, 1, 1, 1])
                    
                    with header_col1:
                        st.markdown(f"### {df['Item_Name'].iloc[0]}")
                        st.caption(f"Item ID: {df['Item_ID'].iloc[0]}")
                    with header_col2:
                        current_price = df['Gold_Price'].iloc[-1]
                        st.metric("Market Value", f"{current_price:,.2f}g")
                    with header_col3:
                        start_price = df['Gold_Price'].iloc[0]
                        trend = ((current_price - start_price) / start_price) * 100 if start_price else 0
                        st.metric("Trend", f"{trend:+.2f}%")
                    with header_col4:
                        current_qty = df['Quantity'].iloc[-1]
                        st.metric("Available Quantity", f"{current_qty:,}")
                    
                    st.divider()

                    # Main Dashboard Layout
                    main_col, side_col = st.columns([3, 1])
                    
                    with main_col:
                        # Line Chart: Price History
                        st.subheader("Price & Quantity History")
                        fig_price = px.line(
                            df, x='timestamp', y='Gold_Price', 
                            color_discrete_sequence=['#2B4C59'],
                            labels={'timestamp': '', 'Gold_Price': 'Gold'}
                        )
                        fig_price.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0))
                        st.plotly_chart(fig_price, use_container_width=True)
                        
                        # Heatmap: Price by Day/Hour
                        st.subheader("Price Heatmap")
                        heatmap_df = df.groupby(['Day_of_Week', 'Hour'])['Gold_Price'].mean().reset_index()
                        fig_heat = go.Figure(data=go.Heatmap(
                            z=heatmap_df['Gold_Price'],
                            x=heatmap_df['Hour'],
                            y=heatmap_df['Day_of_Week'],
                            colorscale=[[0, '#F2EDE4'], [1, '#2B4C59']]
                        ))
                        fig_heat.update_yaxes(categoryorder='array', categoryarray=day_order[::-1])
                        fig_heat.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0))
                        st.plotly_chart(fig_heat, use_container_width=True)

                    with side_col:
                        # Right Sidebar Stats
                        st.subheader("Auction Stats")
                        st.markdown(f"""
                        **Min Buyout:** {df['Gold_Price'].min():,.2f}g  
                        **Max Buyout:** {df['Gold_Price'].max():,.2f}g  
                        **Average Price:** {df['Gold_Price'].mean():,.2f}g  
                        **Total Scans:** {len(df):,}  
                        """)
                else:
                    st.warning("No historical data found for that item.")
                    
            except Exception as e:
                st.error(f"Query Failed: {e}")
