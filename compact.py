import os
import io
import json
import numpy as np
import pandas as pd
import boto3
import duckdb
from datetime import datetime, timedelta
from huggingface_hub import HfApi

# --- CONFIGURATION ---
REALM_IDS = [73, 96, 118] 
B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

B2_BUCKET_NAME = "wow-market-data-lake"
HF_REPO_ID = "bosco277/wow-market-data-lake" # UPDATE THIS

# Connect to Backblaze B2 (Primary Hot Storage)
s3 = boto3.client(
    "s3",
    endpoint_url="https://s3.us-east-005.backblazeb2.com",
    aws_access_key_id=B2_KEY_ID,
    aws_secret_access_key=B2_APP_KEY
)
# Connect to Hugging Face (Secondary Cold Storage)
hf_api = HfApi(token=HF_TOKEN)

def _to_jsonable(obj):
    """Converts pandas/pyarrow nested values (dicts, numpy arrays, numpy scalars) into plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def item_to_json(item):
    """Serializes the full Blizzard item struct (id, context, bonus_lists, modifiers, pet fields...) to a
    stable JSON string so compaction no longer throws the variant data away."""
    if not isinstance(item, dict):
        return None
    return json.dumps(_to_jsonable(item), sort_keys=True, separators=(",", ":"))

def compact_folder(folder_prefix):
    """Merges all hourly files in a folder, drops exact duplicates, and deletes the originals."""
    # 1. Find all files in yesterday's folder on B2
    response = s3.list_objects_v2(Bucket=B2_BUCKET_NAME, Prefix=folder_prefix)
    if 'Contents' not in response:
        print(f"No files found in {folder_prefix}")
        return

    file_keys = [obj['Key'] for obj in response['Contents'] if 'hour_' in obj['Key'] and obj['Key'].endswith('.parquet')]
    
    if not file_keys:
        print(f"No hourly files to compact in {folder_prefix}")
        return

    print(f"Compacting {len(file_keys)} files in {folder_prefix}...")

    # 2. Download from B2 and Merge
    dataframes = []
    for key in file_keys:
        obj = s3.get_object(Bucket=B2_BUCKET_NAME, Key=key)
        df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # 3. Deduplicate 
    original_count = len(merged_df)
    merged_df['clean_item_id'] = merged_df['item'].apply(lambda x: x.get('id') if isinstance(x, dict) else x)
    # STOPGAP: keep the full item struct as JSON (bonus_lists, modifiers, context...) and record which
    # folder (commodities / realm_N) each row came from, instead of discarding both.
    merged_df['item_json'] = merged_df['item'].apply(item_to_json)
    merged_df['source'] = folder_prefix.split('/')[0]
    merged_df = merged_df.drop(columns=['item'])
    merged_df = merged_df.drop_duplicates().reset_index(drop=True)
    
    print(f"Reduced from {original_count} rows to {len(merged_df)} unique records.")

    # 4. Upload Daily File
    daily_filename = f"{folder_prefix}daily.parquet"
    parquet_buffer = io.BytesIO()
    merged_df.to_parquet(parquet_buffer, index=False, compression="snappy")
    parquet_bytes = parquet_buffer.getvalue()
    
    # Upload to B2 (Primary)
    s3.upload_fileobj(io.BytesIO(parquet_bytes), B2_BUCKET_NAME, daily_filename)
    
    # Upload to HF (Backup)
    try:
        hf_api.upload_file(
            path_or_fileobj=io.BytesIO(parquet_bytes),
            path_in_repo=daily_filename,
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )
    except Exception as e:
        print(f"HF Upload Warning: {e}")

    # 5. Delete the Hourly Files
    for key in file_keys:
        s3.delete_object(Bucket=B2_BUCKET_NAME, Key=key)
        try:
            hf_api.delete_file(path_in_repo=key, repo_id=HF_REPO_ID, repo_type="dataset")
        except Exception:
            pass
            
    print(f"Successfully compacted {folder_prefix}")


def generate_dashboard_payloads():
    """Fetches the last 7 days of daily parquets from B2, calculates VWAP, and exports to B2 and HF."""
    print("Generating Dashboard Payloads...")
    
    dataframes = []
    folders_to_check = ["commodities/"] + [f"realm_{r_id}/" for r_id in REALM_IDS]
    
    # Fetch the daily.parquet files from the last 7 days from B2
    for i in range(1, 8):
        d = datetime.utcnow() - timedelta(days=i)
        date_path = f"year={d.year}/month={d.month:02d}/day={d.day:02d}/"
        
        for folder in folders_to_check:
            key = f"{folder}{date_path}daily.parquet"
            try:
                obj = s3.get_object(Bucket=B2_BUCKET_NAME, Key=key)
                df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
                dataframes.append(df)
            except Exception:
                pass 
                
    if not dataframes:
        print("No daily files found for the last 7 days in B2. Skipping dashboard generation.")
        return
        
    raw_df = pd.concat(dataframes, ignore_index=True)
    
    if 'unit_price' in raw_df.columns and 'buyout' in raw_df.columns:
        raw_df['gold_price'] = raw_df['unit_price'].fillna(raw_df['buyout']) / 10000
    elif 'unit_price' in raw_df.columns:
        raw_df['gold_price'] = raw_df['unit_price'] / 10000
    else:
        raw_df['gold_price'] = raw_df['buyout'] / 10000
        
    raw_df['item_id'] = raw_df['clean_item_id']
    micro_lake_df = raw_df[['item_id', 'scan_timestamp', 'gold_price', 'quantity']]
    
    con = duckdb.connect(database=':memory:')
    con.register('micro_lake_df', micro_lake_df)
    
    vwap_query = """
    WITH HourlyVolume AS (
        SELECT 
            item_id,
            scan_timestamp,
            MIN(gold_price) as floor_price,
            SUM(quantity) as volume
        FROM micro_lake_df
        GROUP BY item_id, scan_timestamp
    ),
    RollingVelocity AS (
        SELECT 
            item_id,
            scan_timestamp,
            floor_price,
            volume,
            SUM(volume) OVER (
                PARTITION BY item_id 
                ORDER BY scan_timestamp 
                RANGE BETWEEN INTERVAL 24 HOURS PRECEDING AND CURRENT ROW
            ) as rolling_24h_sales
        FROM HourlyVolume
    )
    SELECT 
        item_id,
        scan_timestamp,
        floor_price,   -- <--- ADD THIS EXACT LINE HERE
        rolling_24h_sales as target_volume,
        SUM(floor_price * volume) OVER (
            PARTITION BY item_id 
            ORDER BY scan_timestamp 
            RANGE BETWEEN INTERVAL 24 HOURS PRECEDING AND CURRENT ROW
        ) / NULLIF(rolling_24h_sales, 0) as liquidity_adjusted_vwap
    FROM RollingVelocity
    WHERE rolling_24h_sales > 0
    """
    vwap_df = con.execute(vwap_query).df()
    
    latest_targets = vwap_df.sort_values('scan_timestamp').groupby('item_id').tail(1).copy()
    latest_targets['buy_threshold'] = latest_targets['liquidity_adjusted_vwap'] * 0.90
    latest_targets['sell_threshold'] = latest_targets['liquidity_adjusted_vwap'] * 0.98
    
    # Standardized upload targeting B2 as primary, HF as secondary
    def export_payload(df, filename):
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, compression="snappy")
        parquet_bytes = buf.getvalue()
        
        # Upload to B2 (Hot Storage)
        s3.upload_fileobj(io.BytesIO(parquet_bytes), B2_BUCKET_NAME, filename)
        
        # Upload to HF (Cold Storage Backup)
        try:
            hf_api.upload_file(
                path_or_fileobj=io.BytesIO(parquet_bytes),
                path_in_repo=filename,
                repo_id=HF_REPO_ID,
                repo_type="dataset"
            )
        except Exception as e:
            print(f"HF Upload Warning for {filename}: {e}")
            
    export_payload(latest_targets, "dashboard_screener.parquet")
    export_payload(vwap_df, "dashboard_charts.parquet")
    print("Successfully generated and uploaded Dashboard Payloads to B2 and Hugging Face.")

def main():
    yesterday = datetime.utcnow() - timedelta(days=1)
    date_path = f"year={yesterday.year}/month={yesterday.month:02d}/day={yesterday.day:02d}/"

    folders_to_check = ["commodities/"] + [f"realm_{r_id}/" for r_id in REALM_IDS]

    for folder in folders_to_check:
        prefix = f"{folder}{date_path}"
        compact_folder(prefix)
        
    generate_dashboard_payloads()

if __name__ == "__main__":
    main()
