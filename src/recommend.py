# recommend.py
import joblib
import logging
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("recommend.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

@st.cache_data
def load_data():
    """Loads the preprocessed data from disk and caches it."""
    logging.info("🔁 Loading data from disk...")
    try:
        # --- KEY CHANGE: Loading new files ---
        df = joblib.load('df_full_cleaned.pkl')
        tfidf_matrix = joblib.load('tfidf_matrix_full.pkl')
        logging.info("✅ Data loaded successfully.")
        return df, tfidf_matrix
    except FileNotFoundError:
        logging.error("❌ Failed to load required files. Make sure to run the updated preprocess.py first.")
        st.error("Data files not found. Please run the preprocessing script first.")
        return None, None
    except Exception as e:
        logging.error("❌ An unexpected error occurred while loading files: %s", str(e))
        st.error(f"An unexpected error occurred: {e}")
        return None, None

def recommend_songs(df, tfidf_matrix, song_name, top_n=5):
    """Recommends songs by calculating similarity on the fly."""
    if df is None or tfidf_matrix is None:
        return None
        
    logging.info("🎵 Recommending songs for: '%s'", song_name)
    
    try:
        idx = df[df['song'].str.lower() == song_name.lower()].index[0]
    except IndexError:
        logging.warning("⚠️ Song '%s' not found in dataset.", song_name)
        return None
    
    # --- KEY CHANGE: Calculate similarity here ---
    song_vector = tfidf_matrix[idx]
    sim_scores = cosine_similarity(song_vector, tfidf_matrix)[0]
    
    # Enumerate scores and sort them
    sim_scores_enum = list(enumerate(sim_scores))
    sim_scores_sorted = sorted(sim_scores_enum, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    
    song_indices = [i[0] for i in sim_scores_sorted]
    
    logging.info("✅ Top %d recommendations ready.", top_n)
    return df.iloc[song_indices]

def recommend_by_artist(df, tfidf_matrix, artist_name, top_n=5):
    """Recommends songs similar to an artist's style on the fly."""
    if df is None or tfidf_matrix is None:
        return None

    logging.info("🎤 Recommending songs for artist: '%s'", artist_name)
    
    artist_songs = df[df['artist'].str.lower() == artist_name.lower()]
    
    if artist_songs.empty:
        logging.warning("⚠️ Artist '%s' not found in dataset.", artist_name)
        return None
        
    first_song_idx = artist_songs.index[0]

    # --- KEY CHANGE: Calculate similarity here ---
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
            
    logging.info("✅ Top %d artist-based recommendations ready.", top_n)
    return df.iloc[recommendations]