import os
import io
import time
import requests
import pandas as pd
import duckdb
import boto3

# Credentials
BLIZZARD_CLIENT_ID = os.environ.get("BLIZZARD_CLIENT_ID")
BLIZZARD_CLIENT_SECRET = os.environ.get("BLIZZARD_CLIENT_SECRET")
B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
B2_BUCKET_NAME = "wow-market-data-lake"

s3_boto = boto3.client(
    "s3",
    endpoint_url="https://s3.us-east-005.backblazeb2.com",
    aws_access_key_id=B2_KEY_ID,
    aws_secret_access_key=B2_APP_KEY
)

def get_blizzard_token():
    url = "https://oauth.battle.net/token"
    auth = (BLIZZARD_CLIENT_ID, BLIZZARD_CLIENT_SECRET)
    data = {"grant_type": "client_credentials"}
    response = requests.post(url, auth=auth, data=data)
    return response.json()["access_token"]

def get_top_item_ids():
    print("Connecting to Data Lake to find the 100 most common items...")
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"""
        CREATE SECRET b2_secret (
            TYPE S3, KEY_ID '{B2_KEY_ID}', SECRET '{B2_APP_KEY}',
            ENDPOINT 's3.us-east-005.backblazeb2.com', URL_STYLE 'path'
        );
    """)
    
    # Scan the commodities folder for the most frequently posted item IDs
    query = """
        SELECT item.id 
        FROM read_parquet('s3://wow-market-data-lake/commodities/**/*.parquet')
        GROUP BY item.id
        ORDER BY count(*) DESC
        LIMIT 100;
    """
    return con.execute(query).df()['id'].tolist()

def fetch_names_from_blizzard(item_ids, token):
    print(f"Fetching names for {len(item_ids)} items...")
    headers = {"Authorization": f"Bearer {token}"}
    item_data = []
    
    for item_id in item_ids:
        # Notice the static-us namespace! Item definitions live here, not in dynamic-us.
        url = f"https://us.api.blizzard.com/data/wow/item/{item_id}?namespace=static-us&locale=en_US"
        res = requests.get(url, headers=headers)
        
        if res.status_code == 200:
            name = res.json().get("name", "Unknown")
            item_data.append({"id": item_id, "name": name})
            print(f"Found: {item_id} -> {name}")
            
        # Sleep for 0.1 seconds so we don't accidentally trigger Blizzard's rate limits
        time.sleep(0.1) 
        
    return pd.DataFrame(item_data)

def main():
    item_ids = get_top_item_ids()
    token = get_blizzard_token()
    
    df_items = fetch_names_from_blizzard(item_ids, token)
    
    print("\nSaving dictionary to Backblaze...")
    parquet_buffer = io.BytesIO()
    df_items.to_parquet(parquet_buffer, index=False, compression="snappy")
    
    parquet_bytes = parquet_buffer.getvalue()
    s3_boto.upload_fileobj(io.BytesIO(parquet_bytes), B2_BUCKET_NAME, "system/item_dictionary.parquet")
    print("Success! Dictionary saved to system/item_dictionary.parquet")

if __name__ == "__main__":
    main()
