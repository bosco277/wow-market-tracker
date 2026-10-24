import os
import io
import requests
import pandas as pd
import boto3
from datetime import datetime
import botocore.exceptions

BLIZZARD_CLIENT_ID = os.environ.get("BLIZZARD_CLIENT_ID")
BLIZZARD_CLIENT_SECRET = os.environ.get("BLIZZARD_CLIENT_SECRET")
B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
B2_BUCKET_NAME = "wow-market-data-lake"

# Initialize the S3 client globally so all functions can use it
s3 = boto3.client(
    "s3",
    endpoint_url="https://s3.us-west-004.backblazeb2.com", # REMEMBER TO UPDATE THIS
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

def get_last_timestamp(realm_id):
    """Reads the last known Blizzard update time from Backblaze"""
    file_key = f"system/last_update_realm_{realm_id}.txt"
    try:
        response = s3.get_object(Bucket=B2_BUCKET_NAME, Key=file_key)
        return response['Body'].read().decode('utf-8')
    except botocore.exceptions.ClientError as e:
        # If the file doesn't exist yet (like on the very first run), return None
        if e.response['Error']['Code'] == 'NoSuchKey':
            return None
        raise e

def save_new_timestamp(realm_id, timestamp_string):
    """Saves Blizzard's newest update time to Backblaze"""
    file_key = f"system/last_update_realm_{realm_id}.txt"
    s3.put_object(Bucket=B2_BUCKET_NAME, Key=file_key, Body=timestamp_string.encode('utf-8'))

def fetch_auctions(realm_id, token, last_modified_date):
    url = f"https://us.api.blizzard.com/data/wow/connected-realm/{realm_id}/auctions"
    params = {
        "namespace": "dynamic-us",
        "locale": "en_US",
        "access_token": token
    }
    
    headers = {}
    if last_modified_date:
        headers["If-Modified-Since"] = last_modified_date

    response = requests.get(url, params=params, headers=headers)
    
    # 304 means Blizzard says "Nothing has changed since that date!"
    if response.status_code == 304:
        return None, None
        
    response.raise_for_status()
    
    # Grab the new timestamp Blizzard sends back so we can save it for next time
    new_timestamp = response.headers.get("Last-Modified")
    return response.json()["auctions"], new_timestamp

def upload_to_b2(dataframe, filename):
    parquet_buffer = io.BytesIO()
    dataframe.to_parquet(parquet_buffer, index=False, compression="snappy")
    parquet_buffer.seek(0) 
    s3.upload_fileobj(parquet_buffer, B2_BUCKET_NAME, filename)
    print(f"Successfully uploaded: {filename}")

def main():
    print("Starting 30-minute auction check...")
    token = get_blizzard_token()
    realm_id = 11
    
    # 1. Ask Backblaze when we last saw an update
    last_known_time = get_last_timestamp(realm_id)
    print(f"Last known Blizzard update: {last_known_time}")
    
    # 2. Ask Blizzard for data (passing the timestamp)
    auctions_json, new_timestamp = fetch_auctions(realm_id, token, last_known_time)
    
    # 3. If Blizzard returns nothing, shut down cleanly to save runner minutes!
    if auctions_json is None:
        print("No new data from Blizzard. Shutting down early to save compute minutes!")
        return

    # 4. If we got new data, process and upload it
    print("New data found! Converting to Parquet...")
    df = pd.DataFrame(auctions_json)
    now = datetime.utcnow()
    filename = f"realm_{realm_id}/year={now.year}/month={now.month:02d}/day={now.day:02d}/hour_{now.hour:02d}_{now.minute:02d}.parquet"
    
    upload_to_b2(df, filename)
    
    # 5. Save the new timestamp back to Backblaze for the next run
    if new_timestamp:
        save_new_timestamp(realm_id, new_timestamp)
        print(f"Saved new timestamp for next run: {new_timestamp}")

if __name__ == "__main__":
    main()
