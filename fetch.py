import os
import io
import requests
import pandas as pd
import boto3
from datetime import datetime

BLIZZARD_CLIENT_ID = os.environ.get("BLIZZARD_CLIENT_ID")
BLIZZARD_CLIENT_SECRET = os.environ.get("BLIZZARD_CLIENT_SECRET")
B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
B2_BUCKET_NAME = "wow-market-data-lake"

def get_blizzard_token():
    url = "https://oauth.battle.net/token"
    auth = (BLIZZARD_CLIENT_ID, BLIZZARD_CLIENT_SECRET)
    data = {"grant_type": "client_credentials"}
    response = requests.post(url, auth=auth, data=data)
    response.raise_for_status()
    return response.json()["access_token"]

def fetch_auctions(realm_id, token):
    url = f"https://us.api.blizzard.com/data/wow/connected-realm/{realm_id}/auctions"
    params = {
        "namespace": "dynamic-us",
        "locale": "en_US",
        "access_token": token
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()["auctions"]

def upload_to_b2(dataframe, filename):
    parquet_buffer = io.BytesIO()
    dataframe.to_parquet(parquet_buffer, index=False, compression="snappy")
    parquet_buffer.seek(0) 
    
    # IMPORTANT: Update this URL to match your specific B2 bucket endpoint
    s3 = boto3.client(
        "s3",
        endpoint_url="https://s3.us-east-005.backblazeb2.com", 
        aws_access_key_id=B2_KEY_ID,
        aws_secret_access_key=B2_APP_KEY
    )
    s3.upload_fileobj(parquet_buffer, B2_BUCKET_NAME, filename)
    print(f"Successfully uploaded: {filename}")

def main():
    print("Starting hourly auction extraction...")
    token = get_blizzard_token()
    
    # We are testing with Realm 11 to make sure it works first
    realm_id = 11
    auctions_json = fetch_auctions(realm_id, token)
    
    df = pd.DataFrame(auctions_json)
    now = datetime.utcnow()
    
    # Creates the folder partition structure DuckDB loves
    filename = f"realm_{realm_id}/year={now.year}/month={now.month:02d}/day={now.day:02d}/hour_{now.hour:02d}.parquet"
    upload_to_b2(df, filename)

if __name__ == "__main__":
    main()
