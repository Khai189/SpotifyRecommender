import requests
import os

DISCOGS_TOKEN = os.environ.get("DISCOGS_TOKEN")

headers = {
    'User-Agent': 'SpotifyRecommend/1.0 +http://example.com',
    'Authorization': f'Discogs token={DISCOGS_TOKEN}'
}

url = "https://api.discogs.com"
params = {
    'q': 'The Dark Side of the Moon',
    'type': 'release',
    'artist': 'Pink Floyd'
}

response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    print("Authentication successful! Here is your identity:")
    results = response.json()
    print(results)

else:
    print(f"Failed to authenticate. Status code: {response.status_code}")
    print(response.text)