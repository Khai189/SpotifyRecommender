import streamlit as st
import requests
import re
import time
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# --- Configuration ---
API_GATEWAY_URL = "https://4zpovu1pp7.execute-api.us-east-1.amazonaws.com/default/recommend" 

# --- Helper to get secrets safely ---
def get_secret(key):
    """Try to get secret from Streamlit secrets, then os.environ."""
    if key in st.secrets:
        return st.secrets[key]
    return os.environ.get(key)

# Spotify OAuth Settings
SPOTIFY_CLIENT_ID = get_secret("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = get_secret("SPOTIFY_CLIENT_SECRET")
# Default to localhost if not set (for local dev), but allow override in secrets (for cloud)
SPOTIFY_REDIRECT_URI = get_secret("SPOTIFY_REDIRECT_URI") or "http://localhost:8501"

# --- Functions ---

def extract_spotify_id(url_or_id):
    """Extracts the Spotify Track ID from a URL or string."""
    match = re.search(r'track/([a-zA-Z0-9]+)', url_or_id)
    if match:
        return match.group(1)
    return url_or_id

def get_recommendations(spotify_id):
    """Calls the backend API and handles the response."""
    full_url = f"{API_GATEWAY_URL}/{spotify_id}"
    status_placeholder = st.empty()
    status_placeholder.info(f"Calling API: {full_url}")

    try:
        response = requests.get(full_url, timeout=20)
        response.raise_for_status()
        
        data = response.json()

        if "message" in data and "starting up" in data["message"]:
            st.warning(data["message"])
            with st.spinner("Waiting for the API to warm up... This can take up to 90 seconds."):
                time.sleep(90)
            st.info("Retrying API call...")
            response = requests.get(full_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            status_placeholder.empty()

        return data

    except requests.exceptions.RequestException as e:
        st.error(f"Sorry, this song isn't currently in our database. Please try another.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def get_spotify_auth_manager():
    """Creates a SpotifyOAuth manager."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        return None
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope="user-read-currently-playing user-read-playback-state",
        cache_path=None # Disable file cache for cloud compatibility
    )

# --- UI ---

st.set_page_config(page_title="Spotify Recommender", layout="wide")

st.title("🎵 Spotify Song Recommender")

auth_manager = get_spotify_auth_manager()
sp = None

if auth_manager:
    # Check for cached token in session state
    if "token_info" not in st.session_state:
        # Try to get token from URL code
        try:
            code = st.query_params.get("code")
            if code:
                st.session_state["token_info"] = auth_manager.get_access_token(code)
                # Clear query params to clean URL
                st.query_params.clear()
        except Exception as e:
            st.error(f"Auth Error: {e}")

    if "token_info" in st.session_state:
        sp = spotipy.Spotify(auth=st.session_state["token_info"]["access_token"])
        # Show disconnect button in sidebar only
        if st.sidebar.button("Disconnect Spotify"):
            del st.session_state["token_info"]
            st.rerun()

# --- Main Content ---

st.write("Enter a Spotify song URL, Track ID, or use your current playback.")

# Input Form
with st.form(key="recommendation_form"):
    spotify_input = st.text_input(
        label="Spotify Song URL or Track ID",
        placeholder="e.g., https://open.spotify.com/track/5TRPicyLGbAF2LGBFbHGvO"
    )
    submit_button = st.form_submit_button(label="Get Recommendations")

st.write("---")

# Spotify Integration Section (Below Form)
if sp:
    # User is logged in
    st.write("Or use what you're listening to right now:")
    if st.button("🎧 Use Current Song"):
        try:
            current_track = sp.current_user_playing_track()
            if current_track and current_track.get("item"):
                track_item = current_track["item"]
                track_name = track_item["name"]
                artist_name = track_item["artists"][0]["name"]
                track_id = track_item["id"]
                
                st.info(f"Detected: **{track_name}** by *{artist_name}*")
                spotify_input = track_id 
                submit_button = True # Force submit
            else:
                st.warning("No song currently playing on your Spotify.")
        except Exception as e:
            st.error(f"Error fetching current song: {e}")
            if "token_info" in st.session_state:
                del st.session_state["token_info"]
                st.rerun()
else:
    # User is NOT logged in
    if auth_manager:
        st.write("Want us to custom recommend based on whatever you're listening to?")
        auth_url = auth_manager.get_authorize_url()
        st.link_button("Connect with Spotify", auth_url)
    else:
        st.warning("Spotify credentials not configured.")

# --- Logic ---

if submit_button and spotify_input:
    spotify_id = extract_spotify_id(spotify_input)
    
    if not spotify_id:
        st.error("Please enter a valid Spotify URL or Track ID.")
    else:
        st.write(f"Fetching recommendations for Spotify ID: `{spotify_id}`")

        with st.spinner("Contacting the recommendation engine..."):
            recommendations = get_recommendations(spotify_id)

        if recommendations:
            if "error" in recommendations:
                st.error("Sorry, your inputted Spotify URL is either not valid or isn't currently supported by our database.")
            else:
                st.success("Here's the 20 songs we think you'll like best!")
                st.info("Songs are ranked with a progress bar, the closer it is to full the better you may like the song")
                st.write("---")
                
                cols = st.columns(4)
                for i, track in enumerate(recommendations):
                    with cols[i % 4]:
                        st.markdown(f"**{i+1}. {track['track_name']}**")
                        st.markdown(f"_{track['artist_name']}_")
                        st.markdown(f"[Listen on Spotify]({track['spotify_url']})")
                        progress_value = 1 - (track['distance'] / 1500)
                        st.progress(max(0.0, min(progress_value, 1.0)))
                        st.markdown("---")

st.sidebar.header("About")
st.sidebar.info(
    "This app uses a machine learning model to find sonically similar songs. "
    "The app aims for songs that sound similar rather than generic genre similarity"
)
