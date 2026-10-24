import os
import io
import requests
import pandas as pd
import boto3
from datetime import datetime
import botocore.exceptions

# --- CONFIGURATION ---
# Replace these with the 5 Realm IDs you pulled from the get_realms script
REALM_IDS = [73, 96, 118] 

BLIZZARD_CLIENT_ID = os.environ.get("BLIZZARD_CLIENT_ID")
BLIZZARD_CLIENT_SECRET = os.environ.get("BLIZZARD_CLIENT_SECRET")
B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
B2_BUCKET_NAME = "wow-market-data-lake"

# Initialize the S3 client globally
s3 = boto3.client(
    "s3",
    endpoint_url="https://s3.us-west-004.backblazeb2.com", # UPDATE THIS TO YOUR B2 ENDPOINT
    aws_access_key_id=B2_KEY_ID,
    aws_secret_access_key=B2_APP_KEY
)

def get_blizzard_token():
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
    """Fetches data using the secure Header authorization and If-Modified-Since logic."""
    params = {
        "namespace": "dynamic-us",
        "locale": "en_US"
    }
    
    # Securely pass the token via Bearer Authorization
    headers = {"Authorization": f"Bearer {token}"}
    
    # If we have checked before, ask Blizzard for only new data
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
    """Converts the JSON to Parquet and uploads it to B2 with partition folders."""
    df = pd.DataFrame(auctions_json)
    now = datetime.utcnow()
    
    # Creates partition folders: realm_11/year=2026/month=10/day=24/hour_09_30.parquet
    filename = f"{folder_name}/year={now.year}/month={now.month:02d}/day={now.day:02d}/hour_{now.hour:02d}_{now.minute:02d}.parquet"
    
    parquet_buffer = io.BytesIO()
    df.to_parquet(parquet_buffer, index=False, compression="snappy")
    parquet_buffer.seek(0) 
    
    s3.upload_fileobj(parquet_buffer, B2_BUCKET_NAME, filename)
    print(f"Successfully uploaded: {filename}")

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

    # 2. Loop Through the 5 Target Realms
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
