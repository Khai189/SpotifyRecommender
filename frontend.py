import streamlit as st
import requests
import re
import time
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth

API_GATEWAY_URL = "https://4zpovu1pp7.execute-api.us-east-1.amazonaws.com/default" 

def get_secret(key):
    if key in st.secrets:
        return st.secrets[key]
    return os.environ.get(key)

SPOTIFY_CLIENT_ID = get_secret("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = get_secret("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = get_secret("SPOTIFY_REDIRECT_URI") or "http://localhost:8501"

def extract_spotify_id(url_or_id):
    match = re.search(r'track/([a-zA-Z0-9]+)', url_or_id)
    if match:
        return match.group(1)
    return url_or_id

def search_database(query):
    if not query or len(query) < 2: return []
    try:
        url = f"{API_GATEWAY_URL}/search"
        response = requests.get(url, params={"q": query}, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Search failed: {e}")
    return []

def get_recommendations(spotify_id):
    full_url = f"{API_GATEWAY_URL}/recommend/{spotify_id}"
    status_placeholder = st.empty()
    status_placeholder.info(f"Calling API: {full_url}")

    try:
        response = requests.get(full_url, timeout=20)
        response.raise_for_status()
        data = response.json()
        status_placeholder.empty() 
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Sorry, this song is not yet in our database")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def get_spotify_auth_manager():
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        return None
    scope = "user-read-currently-playing user-read-playback-state user-modify-playback-state"
    return SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=SPOTIFY_REDIRECT_URI,
        scope=scope,
        cache_path=None
    )

st.set_page_config(page_title="Spotify Recommender", layout="wide")
st.title("🎵 Spotify Song Recommender")

auth_manager = get_spotify_auth_manager()
sp = None

if auth_manager:
    if "token_info" not in st.session_state:
        try:
            code = st.query_params.get("code")
            if code:
                st.session_state["token_info"] = auth_manager.get_access_token(code)
                st.query_params.clear()
        except Exception as e:
            st.error(f"Auth Error: {e}")

    if "token_info" in st.session_state:
        now = int(time.time())
        is_token_expired = st.session_state["token_info"]["expires_at"] - now < 60
        if is_token_expired:
            st.session_state["token_info"] = auth_manager.refresh_access_token(st.session_state["token_info"]["refresh_token"])
        
        sp = spotipy.Spotify(auth=st.session_state["token_info"]["access_token"])
        if st.sidebar.button("Disconnect Spotify"):
            del st.session_state["token_info"]
            st.rerun()

st.write("Enter a Spotify song URL, Track ID, or use your current playback.")

tab1, tab2, tab3 = st.tabs(["🔍 Search Database", "🔗 Paste URL", "🎧 Current Song"])

selected_spotify_id = None

with tab1:
    st.write("Search for a song already in our database.")
    st.caption("Type a song or artist name and press Enter to see matches.")
    search_query = st.text_input("Song or Artist Name", placeholder="e.g. Taylor Swift")
    
    if search_query:
        results = search_database(search_query)
        if results:
            options = {f"{r['track_name']} - {r['artist_name']}": r['spotify_track_id'] for r in results}
            selected_option = st.selectbox("Select a song:", list(options.keys()))
            
            if st.button("Get Recommendations from Search"):
                selected_spotify_id = options[selected_option]
        else:
            st.info("No matches found in our database.")

with tab2:
    st.write("Paste a Spotify Link or Track ID.")
    st.caption("Copy the link from Spotify (Share -> Copy Link) and paste it here.")
    url_input = st.text_input("Spotify URL", placeholder="https://open.spotify.com/track/...")
    if st.button("Get Recommendations from URL"):
        selected_spotify_id = extract_spotify_id(url_input)

with tab3:
    if sp:
        st.write("Use what you're listening to right now.")
        st.caption("We'll detect your currently playing song on Spotify.")
        if st.button("Use Current Song"):
            try:
                current_track = sp.current_user_playing_track()
                if current_track and current_track.get("item"):
                    track_item = current_track["item"]
                    st.info(f"Detected: **{track_item['name']}** by *{track_item['artists'][0]['name']}*")
                    selected_spotify_id = track_item["id"]
                else:
                    st.warning("No song currently playing.")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        if auth_manager:
            st.write("Connect Spotify to use this feature.")
            st.link_button("Connect with Spotify", auth_manager.get_authorize_url())
        else:
            st.warning("Spotify credentials not configured.")

st.write("---")

if selected_spotify_id:
    st.write(f"Fetching recommendations for Spotify ID: `{selected_spotify_id}`")
    with st.spinner("Contacting the recommendation engine..."):
        st.session_state['recommendations'] = get_recommendations(selected_spotify_id)

if 'recommendations' in st.session_state and st.session_state['recommendations']:
    recommendations = st.session_state['recommendations']
    
    if "error" in recommendations:
        st.error("Sorry, that song is not in our database yet.")
    else:
        st.success("Here are the 20 songs we think you'll like best!")
        st.info("Songs are ranked by sonic similarity.")
        st.write("---")
        
        cols = st.columns(4)
        for i, track in enumerate(recommendations):
            with cols[i % 4]:
                st.markdown(f"**{i+1}. {track['track_name']}**")
                st.markdown(f"_{track['artist_name']}_")
                
                if sp:
                    if st.button("Add to Queue", key=track['spotify_track_id']):
                        try:
                            sp.add_to_queue(track['spotify_track_id'])
                            st.toast(f"Added '{track['track_name']}' to queue!")
                        except spotipy.exceptions.SpotifyException as e:
                            if "No active device found" in str(e):
                                st.warning("No active Spotify device found.")
                            else:
                                st.error(f"Error: {e}")
                else:
                    st.markdown(f"[Listen on Spotify]({track['spotify_url']})")

                dist = track['distance']
                scale_factor = 0.05 
                progress_value = 1.0 - (dist / scale_factor)
                final_score = max(0.0, min(progress_value, 1.0))
                
                st.progress(final_score)
                st.caption(f"Similarity Score: {int(final_score * 100)}%")
                st.markdown("---")

st.sidebar.header("About")
st.sidebar.info(
    "This app uses a machine learning model to find sonically similar songs. "
    "The app aims for songs that sound similar rather than generic genre similarity."
)
