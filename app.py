from flask import Flask, jsonify, request
from corpus import Corpus
import os

# Initialize Flask app
app = Flask(__name__)

# Initialize Corpus
TEXT_FOLDER = "./texts"
FREQUENCIES_DIR = "./frequencies"
corpus = Corpus(text_folder=TEXT_FOLDER, frequencies_dir=FREQUENCIES_DIR)

@app.route("/")
def index():
    return jsonify({"message": "Welcome to the Corpus API. Use /analyze or /search endpoints."})

@app.route("/analyze", methods=["GET"])
def analyze_texts():
    """
    Endpoint to return analysis of all texts in the corpus.
    """
    analysis = corpus.analyze_all_texts().to_dict()
    return jsonify(analysis)

@app.route("/list", methods=["GET"])
def list_files():
    """
    Endpoint to return a list with all texts in the corpus.
    """
    texts = corpus.list_files()
    return jsonify(texts)

@app.route("/search", methods=["POST"])
def search_corpus():
    """
    Endpoint to search the corpus with a given query.
    """
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query parameter is required."}), 400

    results = corpus.search(query)
    return jsonify(results.to_dict(orient="records"))

@app.route("/reload", methods=["GET"])
def reload_corpus():
    """
    Endpoint to reload the corpus in case new files are added.
    """
    global corpus
    corpus = Corpus(text_folder=TEXT_FOLDER, frequencies_dir=FREQUENCIES_DIR)
    return jsonify({"message": "Corpus reloaded successfully."})


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(TEXT_FOLDER, exist_ok=True)
    os.makedirs(FREQUENCIES_DIR, exist_ok=True)

    # Run the Flask app
    app.run(debug=True, host="0.0.0.0", port=5000)
