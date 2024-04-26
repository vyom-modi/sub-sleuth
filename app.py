from flask import Flask, render_template, request, flash, redirect, url_for
from math import ceil
import chromadb
import uuid
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# Connect to chroma vector database
CHROMA_DATA_PATH = 'subtitle_chromadb_data'
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = client.get_collection(
    name="subtitle_multi-qa-MiniLM-L6-cos-v1_bert_embeddings_final"
)

# Load CSV data
CSV_FILE_PATH = 'df_sampled.csv'
csv_data = pd.read_csv(CSV_FILE_PATH)

# Initialize temporary data structure for storing search results
temporary_data = {}

def preprocess_text(text):
    # Preprocess the text
    text = re.sub(r'\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def search_query(query):
    preprocessed_query = preprocess_text(query)
    query_embedding = generate_sentence_embeddings([preprocessed_query])[0]
    population = collection.query(query_embeddings=query_embedding.tolist(), n_results=210)
    return population

def generate_sentence_embeddings(subtitles):
    model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    embeddings = model.encode(subtitles, show_progress_bar=True, convert_to_tensor=True)
    if torch.cuda.is_available():
        embeddings = embeddings.cpu()
    embeddings_numpy = embeddings.numpy()
    return embeddings_numpy

def extract_subtitle_info(results):
    subtitle_info = []
    for result in results['metadatas'][0]:
        name = result['name']
        subtitle_info.append({'name': name})
    return subtitle_info

def get_num_from_csv(name):
    match = csv_data[csv_data['name'] == name]
    if not match.empty:
        return match.iloc[0]['num']
    else:
        return None

@app.route('/', methods=["GET", "POST"])
def index():
    global temporary_data
    page = request.args.get('page', 1, type=int)
    per_page = 7  # Number of subtitles per page

    if request.method == "POST":
        query = request.form.get("query")
        if query.strip():
            results = search_query(query)
            subtitle_info = extract_subtitle_info(results)
            subtitles = []
            for info in subtitle_info:
                num = get_num_from_csv(info['name'])
                if num:
                    download_link = f"https://www.opensubtitles.org/en/subtitles/{num}/"
                    subtitles.append({'name': info['name'], 'download_link': download_link})
            
            # Update temporary data structure with all search results for the query
            temporary_data[query] = subtitles

            # Calculate number of pages based on total number of subtitles for the query
            num_subtitles = len(subtitles)
            num_pages = ceil(num_subtitles / per_page)

            # Extract data for the current page
            start = (page - 1) * per_page
            end = start + per_page
            paginated_subtitles = subtitles[start:end]

            return render_template("index.html", subtitles=paginated_subtitles, query=query, page=page, num_pages=num_pages)
        else:
            flash("Please enter a valid query.", "error")
            return redirect(url_for('index'))
    
    query = request.args.get('query')

    if query and query in temporary_data:
        subtitles = temporary_data[query]

        # Calculate number of pages based on total number of subtitles for the query
        num_subtitles = len(subtitles)
        num_pages = ceil(num_subtitles / per_page)

        # Extract data for the current page
        start = (page - 1) * per_page
        end = start + per_page
        paginated_subtitles = subtitles[start:end]

        return render_template("index.html", subtitles=paginated_subtitles, query=query, page=page, num_pages=num_pages)
    
    return render_template("index.html", subtitles=None, query=None, page=page)

if __name__ == '__main__':
    app.run(debug=True)
