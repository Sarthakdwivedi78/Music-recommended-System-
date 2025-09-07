# src/recommend.py
import joblib
import logging
import streamlit as st
import os
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

@st.cache_data
def load_data():
    """Loads the preprocessed data using direct relative paths."""
    logging.info("🔁 Loading data from disk...")
    
    # Use direct paths relative to the project root, which is where Streamlit runs from.
    df_path = 'src/df_full_cleaned.pkl'
    matrix_path = 'src/tfidf_matrix_full.pkl'

    try:
        df = joblib.load(df_path)
        tfidf_matrix = joblib.load(matrix_path)
        logging.info("✅ Data loaded successfully.")
        return df, tfidf_matrix
    except FileNotFoundError as e:
        logging.error(f"❌ Could not find data files. Attempted to load from: {df_path} and {matrix_path}. Error: {e}")
        st.error(f"Data files not found. Please ensure they exist in the 'src' directory in your GitHub repository.")
        
        # Debug block to show what the server sees
        st.write("--- Debug Info ---")
        st.write(f"**Current Working Directory:** `{os.getcwd()}`")
        st.write(f"**Files in Root Directory:** `{os.listdir('.')}`")
        if os.path.exists('src'):
            st.write(f"**Files in 'src' Directory:** `{os.listdir('src')}`")
        else:
            st.write("'src' directory not found.")
        st.write("--- End Debug Info ---")
        
        return None, None
    except Exception as e:
        logging.error("❌ An unexpected error occurred while loading files: %s", str(e))
        st.error(f"An unexpected error occurred: {e}")
        return None, None

def recommend_songs(df, tfidf_matrix, song_name, top_n=5):
    """Recommends songs by calculating similarity on the fly."""
    if df is None or tfidf_matrix is None:
        return None
    try:
        idx = df[df['song'].str.lower() == song_name.lower()].index[0]
    except IndexError:
        return None
    
    song_vector = tfidf_matrix[idx]
    sim_scores = cosine_similarity(song_vector, tfidf_matrix)[0]
    
    sim_scores_enum = list(enumerate(sim_scores))
    sim_scores_sorted = sorted(sim_scores_enum, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    
    song_indices = [i[0] for i in sim_scores_sorted]
    return df.iloc[song_indices]

def recommend_by_artist(df, tfidf_matrix, artist_name, top_n=5):
    """Recommends songs similar to an artist's style on the fly."""
    if df is None or tfidf_matrix is None:
        return None

    artist_songs = df[df['artist'].str.lower() == artist_name.lower()]
    
    if artist_songs.empty:
        return None
        
    first_song_idx = artist_songs.index[0]
    artist_song_vector = tfidf_matrix[first_song_idx]
    sim_scores = cosine_similarity(artist_song_vector, tfidf_matrix)[0]

    sim_scores_enum = list(enumerate(sim_scores))
    sim_scores_sorted = sorted(sim_scores_enum, key=lambda x: x[1], reverse=True)
    
    recommendations = []
    for i, score in sim_scores_sorted:
        if df.iloc[i]['artist'].lower() != artist_name.lower():
            recommendations.append(i)
        if len(recommendations) >= top_n:
            break
            
    return df.iloc[recommendations]
