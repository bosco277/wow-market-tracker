import os
import io
import pandas as pd
import boto3
import numpy as np

# --- CONFIGURATION ---
REALM_IDS = [73, 96, 118]
B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
B2_BUCKET_NAME = "wow-market-data-lake"

s3 = boto3.client("s3", endpoint_url="https://s3.us-east-005.backblazeb2.com", aws_access_key_id=B2_KEY_ID, aws_secret_access_key=B2_APP_KEY)

def apply_tsm_curation(item_df):
    """The exact same TSM logic previously housed in app.py"""
    def process_scan(scan):
        scan = scan.sort_values('Gold_Price')
        total_qty = scan['Quantity'].sum()
        if total_qty == 0: return np.nan
            
        scan['cum_pct'] = scan['Quantity'].cumsum() / total_qty
        trimmed = scan[scan['cum_pct'] <= 0.30].copy()
        if trimmed.empty: trimmed = scan.iloc[[0]]
            
        pct_15_pool = trimmed[trimmed['cum_pct'] <= 0.15]
        if not pct_15_pool.empty:
            trimmed = trimmed[trimmed['Gold_Price'] <= (pct_15_pool['Gold_Price'].max() * 1.20)]
            
        if len(trimmed) > 1:
            mean_p, std_p = trimmed['Gold_Price'].mean(), trimmed['Gold_Price'].std()
            trimmed = trimmed[(trimmed['Gold_Price'] >= mean_p - 1.5 * std_p) & (trimmed['Gold_Price'] <= mean_p + 1.5 * std_p)]
            
        if trimmed.empty: return np.nan
        return np.average(trimmed['Gold_Price'], weights=trimmed['Quantity'])

    scan_baselines = item_df.groupby('timestamp').apply(process_scan, include_groups=False).reset_index(name='scan_baseline').dropna()
    if scan_baselines.empty: return pd.DataFrame()
        
    daily = scan_baselines.set_index('timestamp').resample('1D')['scan_baseline'].mean().dropna().to_frame()
    if not daily.empty:
        daily = daily.loc[daily.index.max() - pd.Timedelta(days=14):]
    daily['tsm_market_value'] = daily['scan_baseline'].ewm(halflife=2, min_periods=1).mean()
    return daily.reset_index()

def build_tsm_master():
    folders = ["commodities"] + [f"realm_{r}" for r in REALM_IDS]
    
    for folder in folders:
        print(f"Building TSM Master for {folder}...")
        
        # 1. Load all historical daily files for the realm
        prefix = f"{folder}/"
        response = s3.list_objects_v2(Bucket=B2_BUCKET_NAME, Prefix=prefix)
        file_keys = [obj['Key'] for obj in response.get('Contents', []) if 'daily.parquet' in obj['Key']]
        
        if not file_keys: continue
            
        dfs = []
        for key in file_keys:
            obj = s3.get_object(Bucket=B2_BUCKET_NAME, Key=key)
            dfs.append(pd.read_parquet(io.BytesIO(obj['Body'].read())))
            
        raw_df = pd.concat(dfs, ignore_index=True)
        
        # Cleanse outliers globally before TSM math
        price_col = 'unit_price' if folder == 'commodities' else 'buyout'
        raw_df['Gold_Price'] = raw_df[price_col] / 10000.0
        raw_df['Quantity'] = raw_df['quantity']
        raw_df['timestamp'] = pd.to_datetime(raw_df['scan_timestamp']).dt.normalize()
        
        # 2. Group by Item ID and apply TSM logic
        tsm_results = []
        # Safely extract the ID from the dictionary into a clean column
        raw_df['clean_item_id'] = raw_df['item'].apply(lambda x: x.get('id') if isinstance(x, dict) else x)

        # Group by the newly extracted column
        for item_id, item_data in raw_df.groupby('clean_item_id'):
            item_tsm = apply_tsm_curation(item_data)
            if not item_tsm.empty:
                item_tsm['Item_ID'] = item_id
                item_tsm['date'] = item_tsm['timestamp'].dt.date
                tsm_results.append(item_tsm[['Item_ID', 'date', 'tsm_market_value']])
                
        if tsm_results:
            final_tsm_df = pd.concat(tsm_results, ignore_index=True)
            
            # 3. Upload the resulting lookup table back to B2
            out_buffer = io.BytesIO()
            final_tsm_df.to_parquet(out_buffer, index=False, compression="snappy")
            s3.upload_fileobj(io.BytesIO(out_buffer.getvalue()), B2_BUCKET_NAME, f"{folder}/system/tsm_master.parquet")
            print(f"-> Saved {len(final_tsm_df)} TSM records for {folder}")

if __name__ == "__main__":
    build_tsm_master()
