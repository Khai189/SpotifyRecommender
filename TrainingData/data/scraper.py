import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os

from scraper_auth import sp
results = sp.current_user_saved_tracks()
for idx, item in enumerate(results['items']):
    track = item['track']
    print(idx, track['artists'][0]['name'], " – ", track['name'])

