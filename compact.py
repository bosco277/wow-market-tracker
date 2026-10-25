import os
import io
import pandas as pd
import boto3
from datetime import datetime, timedelta
from huggingface_hub import HfApi

# --- CONFIGURATION ---
REALM_IDS = [73, 96, 118] 
B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

B2_BUCKET_NAME = "wow-market-data-lake"
HF_REPO_ID = "bosco277/wow-market-data-lake" # UPDATE THIS

s3 = boto3.client(
    "s3",
    endpoint_url="https://s3.us-east-005.backblazeb2.com",
    aws_access_key_id=B2_KEY_ID,
    aws_secret_access_key=B2_APP_KEY
)
hf_api = HfApi(token=HF_TOKEN)

def compact_folder(folder_prefix):
    """Merges all hourly files in a folder, drops duplicates, and deletes the originals."""
    # 1. Find all files in yesterday's folder
    response = s3.list_objects_v2(Bucket=B2_BUCKET_NAME, Prefix=folder_prefix)
    if 'Contents' not in response:
        print(f"No files found in {folder_prefix}")
        return

    file_keys = [obj['Key'] for obj in response['Contents'] if 'hour_' in obj['Key'] and obj['Key'].endswith('.parquet')]
    
    if not file_keys:
        print(f"No hourly files to compact in {folder_prefix}")
        return

    print(f"Compacting {len(file_keys)} files in {folder_prefix}...")

    # 2. Download and Merge
    dataframes = []
    for key in file_keys:
        obj = s3.get_object(Bucket=B2_BUCKET_NAME, Key=key)
        df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # 3. Deduplicate (Keep the most recent update for each auction ID)
    original_count = len(merged_df)
    merged_df.drop_duplicates(subset=['id'], keep='last', inplace=True)
    print(f"Reduced from {original_count} rows to {len(merged_df)} unique auctions.")

    # 4. Upload Daily File
    daily_filename = f"{folder_prefix}daily.parquet"
    parquet_buffer = io.BytesIO()
    merged_df.to_parquet(parquet_buffer, index=False, compression="snappy")
    
    # Extract raw bytes
    parquet_bytes = parquet_buffer.getvalue()
    
    # Upload to B2
    s3.upload_fileobj(io.BytesIO(parquet_bytes), B2_BUCKET_NAME, daily_filename)
    
    # Upload to HF
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
        # Delete from B2
        s3.delete_object(Bucket=B2_BUCKET_NAME, Key=key)
        # Delete from HF
        try:
            hf_api.delete_file(path_in_repo=key, repo_id=HF_REPO_ID, repo_type="dataset")
        except Exception:
            pass
            
    print(f"Successfully compacted {folder_prefix}")

def main():
    # Target yesterday's date
    yesterday = datetime.utcnow() - timedelta(days=1)
    date_path = f"year={yesterday.year}/month={yesterday.month:02d}/day={yesterday.day:02d}/"

    folders_to_check = ["commodities/"] + [f"realm_{r_id}/" for r_id in REALM_IDS]

    for folder in folders_to_check:
        prefix = f"{folder}{date_path}"
        compact_folder(prefix)

if __name__ == "__main__":
    main()
