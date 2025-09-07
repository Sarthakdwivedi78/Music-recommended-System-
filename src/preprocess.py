# preprocess.py
import pandas as pd
import re
import nltk
import joblib
import logging
from pathlib import Path  # Make sure to import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🚀 Starting preprocessing for the full dataset...")

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# Load the FULL dataset using a reliable path
try:
    # Get the directory of the current script
    script_dir = Path(__file__).parent
    # Create the full path to the data file
    data_file_path = script_dir / "spotify_millsongdata.csv"
    
    # Read the CSV using the full path
    df = pd.read_csv(data_file_path)
    logging.info("✅ Full dataset loaded: %d rows", len(df))

except FileNotFoundError:
    logging.error(f"❌ Data file not found at {data_file_path}. Please ensure 'spotify_millsongdata.csv' is in the same directory as this script.")
    # Exit if the file is not found
    exit()
except Exception as e:
    logging.error("❌ Failed to load dataset: %s", str(e))
    raise e

# Drop duplicates and rows with missing lyrics
df = df.drop_duplicates(subset='song').dropna(subset=['text', 'artist'])
df = df.drop(columns=['link'], errors='ignore').reset_index(drop=True)

# Text cleaning
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

logging.info("🧹 Cleaning text for all %d songs...", len(df))
df['cleaned_text'] = df['text'].apply(preprocess_text)
logging.info("✅ Text cleaned.")

# Vectorization
logging.info("🔠 Vectorizing using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])
logging.info("✅ TF-IDF matrix shape: %s", tfidf_matrix.shape)

# Save the necessary components
# Note: Saving files in the same directory as the script
output_dir = Path(__file__).parent
joblib.dump(df, output_dir / 'df_full_cleaned.pkl')
joblib.dump(tfidf_matrix, output_dir / 'tfidf_matrix_full.pkl')
joblib.dump(tfidf, output_dir / 'tfidf_vectorizer.pkl')
logging.info("💾 Data, TF-IDF matrix, and vectorizer saved to disk.")

logging.info("✅ Preprocessing complete.")