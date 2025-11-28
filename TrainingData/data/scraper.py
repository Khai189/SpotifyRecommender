from concurrent.futures import ThreadPoolExecutor, as_completed
import spotify_scraper
import json
from pathlib import Path
from spotify_scraper import SpotifyClient
from spotify_scraper.extractors import (
    TrackExtractor,
    AlbumExtractor,
    ArtistExtractor,
    PlaylistExtractor
)
from spotify_scraper.parsers import json_parser
# Set up our client
client = SpotifyClient()

# Run a test case for a track before we start generating our data
track_url = "https://open.spotify.com/track/61mWefnWQOLf90gepjOCb3"
track = client.get_track_info(track_url)
preview_path = client.download_preview_mp3(track_url,
                                           path=Path("mp3_files"))
artist = track.get('artists', [])[0].get("name")
with open(f"mp3_files/{track["name"]}_{artist}.json", "w") as f:
    f.write(json.dumps(track))
# json_parser.extract_track_data()
