from flask import Flask, jsonify, request, render_template, send_from_directory
from corpus import Corpus
import os

# Initialize Flask App
app = Flask(__name__, static_folder="static", template_folder="templates")

# Set Text and Frequencies Folders
TEXT_FOLDER = "./texts"
FREQUENCIES_DIR = "./frequencies"
corpus = Corpus(text_folder=TEXT_FOLDER, frequencies_dir=FREQUENCIES_DIR)

@app.route("/")
def index():
    """
    Serve the main HTML page.
    """
    return render_template("index.html")

@app.route("/analyze", methods=["GET"])
def analyze_texts():
    """
    Endpoint to analyze all texts in the corpus and return metrics.
    """
    return jsonify(corpus.analyze_all_texts().to_dict())

@app.route("/search", methods=["POST"])
def search_corpus():
    """
    Endpoint to search the corpus with a query and return results.
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
    Endpoint to reload the corpus.
    """
    global corpus
    corpus = Corpus(text_folder=TEXT_FOLDER, frequencies_dir=FREQUENCIES_DIR)
    return jsonify({"message": "Corpus reloaded successfully."})

@app.route("/files", methods=["GET"])
def list_files():
    """
    Endpoint to list all files in the corpus.
    """
    files = corpus.list_files()
    return jsonify(files)

@app.route("/upload", methods=["POST"])
def upload_to_corpus():
    """
    Endpoint to upload new files to the corpus.
    """
    uploaded_files = request.files.getlist("files")
    added_files = []
    skipped_files = []

    for file in uploaded_files:
        if not file.filename.endswith(".txt"):
            skipped_files.append({
                "file": file.filename,
                "reason": "Invalid file format. Only .txt files are supported."
            })
            continue

        file_path = os.path.join(TEXT_FOLDER, file.filename)
        file.save(file_path)
        added_files.append(file.filename)

    # Reload corpus to include new files
    global corpus
    corpus = Corpus(text_folder=TEXT_FOLDER, frequencies_dir=FREQUENCIES_DIR)

    return jsonify({
        "added_files": added_files,
        "skipped_files": skipped_files,
        "message": "Upload complete."
    })

@app.route("/static/<path:filename>")
def serve_static(filename):
    """
    Serve static files such as CSS or JS.
    """
    return send_from_directory(app.static_folder, filename)

if __name__ == "__main__":
    os.makedirs(TEXT_FOLDER, exist_ok=True)
    os.makedirs(FREQUENCIES_DIR, exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
