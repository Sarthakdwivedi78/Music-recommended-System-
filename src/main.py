# app.py
import streamlit as st
from recommend import load_data, recommend_songs, recommend_by_artist

import joblib
import os

# Get the absolute path of the directory the current script (main.py) is in
script_dir = os.path.dirname(os.path.abspath(__file__))

# Join the script's directory with the filename
df_path = os.path.join(script_dir, 'df_full_cleaned.pkl')
vectorizer_path = os.path.join(script_dir, 'tfidf_vectorizer.pkl')

# Load the files using the robust, absolute paths
df = joblib.load(df_path)
vectorizer = joblib.load(vectorizer_path)
# --- 1. Page Configuration ---
st.set_page_config(
    page_title="MeloMix 🎵 - Music Recommender",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Load Data ---
# --- KEY CHANGE: This now returns df and tfidf_matrix ---
df, tfidf_matrix = load_data()

# --- 3. App Styling ---
st.markdown("""
<style>
    /* Main title */
    .stApp > header { background-color: transparent; }
    /* Recommendation card styling */
    .recommendation-card { background-color: #222222; padding: 20px; border-radius: 12px; margin-bottom: 12px; border: 1px solid #444444; transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out; }
    .recommendation-card:hover { transform: translateY(-5px); box-shadow: 0 8px 16px rgba(0,0,0,0.3); }
    .recommendation-card h3 { color: #1DB954; margin-top: 0; font-size: 1.2rem; }
    .recommendation-card p { color: #B3B3B3; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# --- 4. Sidebar ---
with st.sidebar:
    st.image("https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_White.png", width=200)
    st.title("MeloMix Controls")
    recommendation_mode = st.radio("Choose your recommendation mode:", ("Recommend by Song", "Recommend by Artist"))
    st.markdown("---")
    num_recommendations = st.slider("Number of Recommendations", 3, 15, 5)

# --- 5. Main Content ---
st.title("🎶 MeloMix Music Recommender")
st.markdown("Discover new music from over **57,000+ songs** based on your favorite artists and tracks.")

if df is not None:
    if recommendation_mode == "Recommend by Song":
        song_list = sorted(df['song'].dropna().unique())
        selected_song = st.selectbox("🎵 Select a song you like:", song_list, index=None, placeholder="Type or select a song...")
        
        if st.button("🚀 Recommend Songs", use_container_width=True) and selected_song:
            with st.spinner(f"Finding songs similar to '{selected_song}'..."):
                # --- KEY CHANGE: Pass tfidf_matrix instead of cosine_sim ---
                recommendations = recommend_songs(df, tfidf_matrix, selected_song, top_n=num_recommendations)
                if recommendations is not None and not recommendations.empty:
                    st.success(f"Here are your top {num_recommendations} recommendations:")
                    cols = st.columns(3) 
                    for i, (_, row) in enumerate(recommendations.iterrows()):
                        with cols[i % 3]:
                            st.markdown(f'<div class="recommendation-card"><h3>{i+1}. {row["song"]}</h3><p><strong>Artist:</strong> {row["artist"]}</p></div>', unsafe_allow_html=True)
                else:
                    st.warning("Could not find recommendations for that song. Try another one! 🧐")

    elif recommendation_mode == "Recommend by Artist":
        artist_list = sorted(df['artist'].dropna().unique())
        selected_artist = st.selectbox("🎤 Select an artist you like:", artist_list, index=None, placeholder="Type or select an artist...")
        
        if st.button("✨ Recommend by Artist", use_container_width=True) and selected_artist:
            with st.spinner(f"Finding artists with a vibe like '{selected_artist}'..."):
                # --- KEY CHANGE: Pass tfidf_matrix instead of cosine_sim ---
                recommendations = recommend_by_artist(df, tfidf_matrix, selected_artist, top_n=num_recommendations)
                if recommendations is not None and not recommendations.empty:
                    st.success(f"Here are {num_recommendations} songs from artists with a similar vibe:")
                    cols = st.columns(3)
                    for i, (_, row) in enumerate(recommendations.iterrows()):
                        with cols[i % 3]:
                            st.markdown(f'<div class="recommendation-card"><h3>{i+1}. {row["song"]}</h3><p><strong>Artist:</strong> {row["artist"]}</p></div>', unsafe_allow_html=True)
                else:
                    st.warning("Could not find recommendations for that artist. Try another one! 🧐")
else:
    st.error("Data could not be loaded. Please ensure you have run the updated preprocessing script.")
