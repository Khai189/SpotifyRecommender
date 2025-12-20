import streamlit as st
import requests
import re
import time

API_GATEWAY_URL = "https://4zpovu1pp7.execute-api.us-east-1.amazonaws.com/default/recommend" # Your actual API Gateway URL + stage + resource path

def extract_spotify_id(url_or_id):
    """Extracts the Spotify Track ID from a URL or string."""
    match = re.search(r'track/([a-zA-Z0-9]+)', url_or_id)
    if match:
        return match.group(1)
    return url_or_id

def get_recommendations(spotify_id):
    """Calls the backend API and handles the response."""
    full_url = f"{API_GATEWAY_URL}/{spotify_id}"
    st.info(f"Calling API: {full_url}")

    try:
        response = requests.get(full_url, timeout=20) # 20-second timeout for initial call
        response.raise_for_status()
        
        data = response.json()

        if "message" in data and "starting up" in data["message"]:
            st.warning(data["message"])
            with st.spinner("Waiting for the API to warm up... This can take up to 90 seconds."):
                time.sleep(90)
            # Retry after waiting
            st.info("Retrying API call...")
            response = requests.get(full_url, timeout=30)
            response.raise_for_status()
            data = response.json()

        return data

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while calling the API: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

st.set_page_config(page_title="Spotify Recommender", layout="wide")

st.title("🎵 Spotify Song Recommender")
st.write("Enter a Spotify song URL or Track ID to get recommendations based on its audio features.")
st.write("")
with st.form(key="recommendation_form"):
    spotify_input = st.text_input(
        label="Spotify Song URL or Track ID",
        placeholder="e.g., https://open.spotify.com/track/5TRPicyLGbAF2LGBFbHGvO"
    )
    submit_button = st.form_submit_button(label="Get Recommendations")

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
