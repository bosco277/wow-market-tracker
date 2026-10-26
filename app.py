import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
                
                if not macro_df.empty:
                    # Prepare the data and build the Treemap
                    trend_data = prep_treemap_data(macro_df, meta_df)
                    treemap_fig = create_market_treemap(trend_data)
                    
                    # Display the Treemap
                    st.plotly_chart(treemap_fig, use_container_width=True)
                else:
                    st.warning("No data found for this realm/category.")
            except Exception as e:
                st.error(f"Macro Query Failed: {e}")
