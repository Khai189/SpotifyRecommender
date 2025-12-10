import spotipy
from spotipy.oauth2 import SpotifyOAuth
import discogs_client
from scraper_auth import *
import os

results = sp.current_user_saved_tracks()
for idx, item in enumerate(results['items']):
    track = item['track']
    print(idx, track['artists'][0]['name'], " – ", track['name'])


user_agent = "spotify_discogs_recommend"

discogsclient = discogs_client.Client(user_agent)

discogsclient.set_consumer_key(consumer_key, consumer_secret)
token, secret, url = discogsclient.get_authorize_url()

print(" == Request Token == ")
print(f"    * oauth_token        = {token}")
print(f"    * oauth_token_secret = {secret}")
print()

print(f"Please browse to the following URL {url}")
accepted = "n"
while accepted.lower() == "n":
    print()
    accepted = input(f"Have you authorized me at {url} [y/n] :")

oauth_verifier = input("Verification code : ")

access_token, access_secret = discogsclient.get_access_token(oauth_verifier)
user = discogsclient.identity()
search_results = discogsclient.search(
    "House For All", type="release", artist="Blunted Dummies"
)

for release in search_results:
    print(f"\n\t== discogs-id {release.id} ==")
    print(f'\tArtist\t: {", ".join(artist.name for artist in release.artists)}')
    print(f"\tTitle\t: {release.title}")
    print(f"\tYear\t: {release.year}")
    print(f'\tLabels\t: {", ".join(label.name for label in release.labels)}')