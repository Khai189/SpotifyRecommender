import requests
from scraper_auth import *
import json
API_KEY = api_key_last_fm
USER_AGENT = os.environ.get('USER_AGENT')

def connect_last_fm(user_agent, api_key):
    headers = {
        'user-agent': user_agent
    }

    payload = {
        'api_key': api_key,
        'method': 'chart.getTopTracks',
        'format': 'json'
    }
    r = requests.get('https://ws.audioscrobbler.com/2.0/', headers=headers, params=payload)
    return r

results_last_fm = connect_last_fm(USER_AGENT, API_KEY)

def jprint(obj):
    print(json.dumps(obj, sort_keys=True, indent=4))

def lastfm_get(user_agent, api_key, payload):
    headers = {'user-agent': user_agent}
    url = 'https://ws.audioscrobbler.com/2.0/'

    payload['api_key'] = api_key
    payload['format'] = 'json'

    response = requests.get(url, headers=headers, params=payload)
    return response

r = lastfm_get(
    user_agent=USER_AGENT,
    api_key=API_KEY,
    payload={
    'method': 'chart.getTopTracks'
})
def get_top_50_track_artists(results):
    artists = []
    for i in range(50):
        artist = results.json()["tracks"]["track"][i]["artist"]["name"]
        if artist not in artists:
            artists.append(artist)
    return artists

print(get_top_50_track_artists(r))