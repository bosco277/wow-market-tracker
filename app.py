import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json

# ---------------------------------------------------------
# HELPER FUNCTIONS FOR TREEMAP
# ---------------------------------------------------------
@st.cache_data
def load_metadata():
    # Load the local JSON file containing the WoW items database
    try:
        with open('items.json', 'r', encoding='utf-8') as f:
            item_data = json.load(f)
        df_meta = pd.DataFrame(item_data)
        # Rename columns to match the auction dataframe
        df_meta = df_meta.rename(columns={'itemId': 'item_id', 'class': 'item_class', 'subclass': 'item_subclass'})
        return df_meta
    except FileNotFoundError:
        st.error("items.json not found. Please upload it to your repository.")
        return pd.DataFrame(columns=['item_id', 'item_class', 'item_subclass'])

def prep_treemap_data(auction_df, meta_df):
    # Merge live auction data with your local items.json metadata
    df = pd.merge(auction_df, meta_df, left_on='Item_ID', right_on='item_id', how='left')
    
    # Calculate current average price and total volume per item
    current_stats = df.groupby(['Item_Name', 'item_class', 'item_subclass']).agg(
        current_price=('Gold_Price', 'mean'),
        total_volume=('Quantity', 'sum') 
    ).reset_index()
    
    # Calculate a historical baseline (for the trend calculation)
    historical_stats = df.groupby('Item_Name').agg(
        hist_price=('Gold_Price', 'mean')
    ).reset_index()
    
    # Merge and calculate the percentage trend
    trend_df = pd.merge(current_stats, historical_stats, on='Item_Name')
    
    # Avoid division by zero
    trend_df['hist_price'] = trend_df['hist_price'].replace(0, np.nan)
    trend_df['price_trend_pct'] = ((trend_df['current_price'] - trend_df['hist_price']) / trend_df['hist_price']) * 100
    trend_df['price_trend_pct'] = trend_df['price_trend_pct'].fillna(0)
    
    # Fill missing metadata with 'Other' so the chart doesn't break
    trend_df.fillna({'item_class': 'Other', 'item_subclass': 'Other'}, inplace=True)
    
    return trend_df

def create_market_treemap(trend_df):
    fig = px.treemap(
        trend_df,
        path=[px.Constant("All Categories"), 'item_class', 'item_subclass', 'Item_Name'],
        values='total_volume',
        color='price_trend_pct',
        color_continuous_scale=['#b22222', '#2f4f4f', '#228b22'], 
        color_continuous_midpoint=0,
    )
    
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Volume: %{value}<br>Trend: %{color:.1f}%<extra></extra>"
    )
    fig.update_layout(margin=dict(t=30, l=10, r=10, b=10), paper_bgcolor='rgba(0,0,0,0)')
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
    search_id = st.text_input("Search Item ID or Name", placeholder="e.g., Linen Cloth or 2589")
    search_term = search_id.strip()
    
    if st.button(f"Fetch Item Data for {selected_display_name}"):
        if not search_term:
            st.warning("Please enter an Item ID or Item Name in the search box to view its historical dashboard.")
        else:
            with st.spinner(f"Compiling historical market data for {search_term}..."):
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
                            st.metric("Market Value", f"{current_price:,.2f}g")
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
                            
                            # 1. Aggregate data: Min Price and Total Quantity per scan
                            agg_df = df.groupby('timestamp').agg(
                                Min_Price=('Gold_Price', 'min'),
                                Total_Quantity=('Quantity', 'sum')
                            ).reset_index()

                            # 2. Setup dual-axis figure
                            fig_price = make_subplots(specs=[[{"secondary_y": True}]])

                            # 3. Add Total Quantity Bars (Accent Color)
                            fig_price.add_trace(
                                go.Bar(
                                    x=agg_df['timestamp'], 
                                    y=agg_df['Total_Quantity'], 
                                    name='Total Quantity',
                                    marker_color='#C98A4B', # Warm amber for supporting bars
                                    opacity=0.4
                                ),
                                secondary_y=True,
                            )

                            # 4. Add Minimum Price Line (Primary Color)
                            fig_price.add_trace(
                                go.Scatter(
                                    x=agg_df['timestamp'], 
                                    y=agg_df['Min_Price'], 
                                    name='Min Price',
                                    mode='lines',
                                    line=dict(color='#2B4C59', width=2) # Deep slate blue
                                ),
                                secondary_y=False,
                            )

                            # 5. Calculate and Add Trend Line
                            if len(agg_df) > 1:
                                x_numeric = pd.to_numeric(agg_df['timestamp'])
                                z = np.polyfit(x_numeric, agg_df['Min_Price'], 1)
                                p = np.poly1d(z)
                                
                                fig_price.add_trace(
                                    go.Scatter(
                                        x=agg_df['timestamp'], 
                                        y=p(x_numeric), 
                                        name='Price Trend',
                                        mode='lines',
                                        line=dict(color='#2B4C59', width=2, dash='dot') # Deep slate blue for main trend line
                                    ),
                                    secondary_y=False,
                                )

                            # 6. Layout adjustments
                            # Force bars to sit at the bottom by expanding the secondary Y-axis range
                            max_qty = agg_df['Total_Quantity'].max()
                            fig_price.update_yaxes(title_text="Gold", secondary_y=False, rangemode='tozero')
                            fig_price.update_yaxes(showgrid=False, secondary_y=True, range=[0, max_qty * 4], showticklabels=False)

                            fig_price.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)', 
                                paper_bgcolor='rgba(0,0,0,0)', 
                                margin=dict(l=0, r=0, t=10, b=0),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                            )
                            
                            st.plotly_chart(fig_price, use_container_width=True)
                            
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


# ==========================================
# TAB 2: OVERALL MARKET TRENDS (TREEMAP)
# ==========================================
with tab2:
    st.subheader(f"Market Trends for {selected_display_name}")
    st.write("Click on a category block to drill down into specific subcategories and items. Color indicates price trend, while block size indicates available quantity.")
    
    if st.button(f"Generate Heatmap for {selected_display_name}"):
        with st.spinner("Aggregating overall market data. This may take a moment..."):
            
            # Query the entire realm/category without the WHERE filter
            macro_query = f"""
                SELECT 
                    COALESCE(dict.name, 'Unknown Item') as Item_Name,
                    auc.item.id as Item_ID, 
                    (auc.{price_col} / 10000.0) as Gold_Price, 
                    auc.quantity as Quantity,
                    COALESCE(auc.scan_timestamp, make_timestamp(year, month, day, 0, 0, 0)) as timestamp
                FROM read_parquet('{s3_path}', hive_partitioning=1, union_by_name=True) as auc
                LEFT JOIN read_parquet('{dict_path}') as dict 
                    ON auc.item.id = dict.id
            """
            try:
                macro_df = db.execute(macro_query).df()
                
def prep_treemap_data(auction_df, meta_df):
    # Merge live auction data with your local items.json metadata
    df = pd.merge(auction_df, meta_df, left_on='Item_ID', right_on='item_id', how='left')
    
    # 1. Identify the most recent scan timestamp
    latest_scan = df['timestamp'].max()
    
    # 2. Separate current data from historical data
    current_df = df[df['timestamp'] == latest_scan]
    historical_df = df[df['timestamp'] < latest_scan]
    
    # 3. Calculate Current Stats (using minimum price to match Tab 1 logic)
    current_stats = current_df.groupby(['Item_Name', 'item_class', 'item_subclass']).agg(
        current_price=('Gold_Price', 'min'),
        total_volume=('Quantity', 'sum') 
    ).reset_index()
    
    # 4. Calculate Historical Baseline
    if not historical_df.empty:
        # Get the minimum price for each historical scan, then average them together
        hist_scan_mins = historical_df.groupby(['Item_Name', 'timestamp']).agg(scan_min=('Gold_Price', 'min')).reset_index()
        historical_stats = hist_scan_mins.groupby('Item_Name').agg(hist_price=('scan_min', 'mean')).reset_index()
    else:
        # Fallback if there is only one scan in the database
        historical_stats = current_stats[['Item_Name', 'current_price']].copy()
        historical_stats.rename(columns={'current_price': 'hist_price'}, inplace=True)
        
    # 5. Merge and calculate the percentage trend
    trend_df = pd.merge(current_stats, historical_stats, on='Item_Name')
    
    # Avoid division by zero
    trend_df['hist_price'] = trend_df['hist_price'].replace(0, np.nan)
    trend_df['price_trend_pct'] = ((trend_df['current_price'] - trend_df['hist_price']) / trend_df['hist_price']) * 100
    trend_df['price_trend_pct'] = trend_df['price_trend_pct'].fillna(0)
    
    # Fill missing metadata with 'Other' so the chart doesn't break
    trend_df.fillna({'item_class': 'Other', 'item_subclass': 'Other'}, inplace=True)
    
    return trend_df
            except Exception as e:
                st.error(f"Macro Query Failed: {e}")
