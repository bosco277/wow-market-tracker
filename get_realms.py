import os
import requests

BLIZZARD_CLIENT_ID = os.environ.get("BLIZZARD_CLIENT_ID")
BLIZZARD_CLIENT_SECRET = os.environ.get("BLIZZARD_CLIENT_SECRET")

def get_blizzard_token():
    url = "https://oauth.battle.net/token"
    auth = (BLIZZARD_CLIENT_ID, BLIZZARD_CLIENT_SECRET)
    data = {"grant_type": "client_credentials"}
    response = requests.post(url, auth=auth, data=data)
    response.raise_for_status()
    return response.json()["access_token"]

def print_realm_list(token):
    index_url = "https://us.api.blizzard.com/data/wow/connected-realm/index"
    
    # 1. The token is removed from params
    params = {"namespace": "dynamic-us", "locale": "en_US"}
    
    # 2. The token is now sent securely in the Authorization Header
    headers = {"Authorization": f"Bearer {token}"}
    
    print("Fetching Connected Realm Index...")
    
    # 3. We pass the headers to the requests.get() function
    index_response = requests.get(index_url, params=params, headers=headers)
    index_response.raise_for_status()
    
    connected_realms = index_response.json()["connected_realms"]
    realm_ids = [link["href"].split("/")[-1].split("?")[0] for link in connected_realms]

    print(f"Found {len(realm_ids)} Connected Realms. Fetching names...\n")

    for r_id in realm_ids:
        details_url = f"https://us.api.blizzard.com/data/wow/connected-realm/{r_id}"
        details_response = requests.get(details_url, params=params, headers=headers)
        
        if details_response.status_code == 200:
            realm_names = [sub["name"] for sub in details_response.json().get("realms", [])]
            print(f"ID {r_id}: {', '.join(realm_names)}")
        else:
             print(f"ID {r_id}: Failed to fetch details.")

if __name__ == "__main__":
    token = get_blizzard_token()
    print_realm_list(token)
