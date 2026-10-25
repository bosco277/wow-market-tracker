import os
import io
import requests
import pandas as pd
import boto3
from datetime import datetime
import botocore.exceptions
from huggingface_hub import HfApi # 1. ADD THIS IMPORT

# --- CONFIGURATION ---
# Replace these with the 5 Realm IDs you pulled from the get_realms script
REALM_IDS = [73, 96, 118] 

# Credentials pulled securely from GitHub Secrets
BLIZZARD_CLIENT_ID = os.environ.get("BLIZZARD_CLIENT_ID")
BLIZZARD_CLIENT_SECRET = os.environ.get("BLIZZARD_CLIENT_SECRET")
B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

B2_BUCKET_NAME = "wow-market-data-lake"
# IMPORTANT: Update with your Hugging Face username!
HF_REPO_ID = "bosco277/wow-market-data-lake" 

# Initialize the S3 client globally (Using your confirmed us-east-005 endpoint)
s3 = boto3.client(
    "s3",
    endpoint_url="https://s3.us-east-005.backblazeb2.com",
    aws_access_key_id=B2_KEY_ID,
    aws_secret_access_key=B2_APP_KEY
)

def get_blizzard_token():
    """Fetches a temporary access token from Blizzard."""
    url = "https://oauth.battle.net/token"
    auth = (BLIZZARD_CLIENT_ID, BLIZZARD_CLIENT_SECRET)
    data = {"grant_type": "client_credentials"}
    response = requests.post(url, auth=auth, data=data)
    response.raise_for_status()
    return response.json()["access_token"]

def get_last_timestamp(tracker_name):
    """Reads the last known update time from Backblaze for a specific dataset."""
    file_key = f"system/last_update_{tracker_name}.txt"
    try:
        response = s3.get_object(Bucket=B2_BUCKET_NAME, Key=file_key)
        return response['Body'].read().decode('utf-8')
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return None
        raise e

def save_new_timestamp(tracker_name, timestamp_string):
    """Saves Blizzard's newest update time to Backblaze."""
    file_key = f"system/last_update_{tracker_name}.txt"
    s3.put_object(Bucket=B2_BUCKET_NAME, Key=file_key, Body=timestamp_string.encode('utf-8'))

def fetch_blizzard_data(url, token, last_modified_date):
    """Fetches data using secure Header authorization and If-Modified-Since logic."""
    params = {
        "namespace": "dynamic-us",
        "locale": "en_US"
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    
    if last_modified_date:
        headers["If-Modified-Since"] = last_modified_date

    response = requests.get(url, params=params, headers=headers)
    
    # 304 code means Blizzard has no new data for this specific endpoint
    if response.status_code == 304:
        return None, None
        
    response.raise_for_status()
    
    new_timestamp = response.headers.get("Last-Modified")
    return response.json().get("auctions", []), new_timestamp

def process_and_upload(auctions_json, folder_name):
    """Converts JSON to Parquet and dual-writes to Backblaze B2 and Hugging Face."""
    df = pd.DataFrame(auctions_json)
    now = datetime.utcnow()
    
    # ADD THIS LINE: Explicitly stamp every row with the exact time
    df['scan_timestamp'] = now 
    
    filename = f"{folder_name}/year={now.year}/month={now.month:02d}/day={now.day:02d}/hour_{now.hour:02d}_{now.minute:02d}.parquet"
    
    # Create the initial buffer and save the Parquet data
    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer, index=False, compression="snappy")
    
    # EXTRACT THE RAW BYTES SO NO LIBRARY CAN CLOSE OUR STREAM
    parquet_bytes = parquet_buffer.getvalue()
    
    # 1. Upload to Backblaze B2 (using a fresh wrapper)
    s3.upload_fileobj(io.BytesIO(parquet_bytes), B2_BUCKET_NAME, filename)
    print(f"Successfully uploaded to B2: {filename}")

    # 2. Upload to Hugging Face (using a fresh wrapper)
    try:
        api = HfApi(token=HF_TOKEN)
        api.upload_file(
            path_or_fileobj=io.BytesIO(parquet_bytes),
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
            repo_type="dataset"
        )
        print(f"Successfully uploaded to Hugging Face: {filename}")
    except Exception as e:
        print(f"Failed to upload to Hugging Face: {e}")

def main():
    print("Starting auction extraction cycle...")
    token = get_blizzard_token()
    
    # 1. Process Regional Commodities First
    print("\n--- Checking Commodities ---")
    commodities_url = "https://us.api.blizzard.com/data/wow/auctions/commodities"
    last_comm_time = get_last_timestamp("commodities")
    
    comm_data, comm_timestamp = fetch_blizzard_data(commodities_url, token, last_comm_time)
    
    if comm_data is None:
        print("No new Commodities data. Skipping.")
    else:
        print(f"New Commodities data found. Processing {len(comm_data)} auctions...")
        process_and_upload(comm_data, "commodities")
        save_new_timestamp("commodities", comm_timestamp)

    # 2. Loop Through the Target Realms
    for realm_id in REALM_IDS:
        print(f"\n--- Checking Realm {realm_id} ---")
        realm_url = f"https://us.api.blizzard.com/data/wow/connected-realm/{realm_id}/auctions"
        
        last_realm_time = get_last_timestamp(f"realm_{realm_id}")
        realm_data, realm_timestamp = fetch_blizzard_data(realm_url, token, last_realm_time)
        
        if realm_data is None:
            print(f"No new data for Realm {realm_id}. Skipping.")
        else:
            print(f"New data found for Realm {realm_id}. Processing {len(realm_data)} auctions...")
            process_and_upload(realm_data, f"realm_{realm_id}")
            save_new_timestamp(f"realm_{realm_id}", realm_timestamp)

    print("\nCycle complete!")

if __name__ == "__main__":
    main()
