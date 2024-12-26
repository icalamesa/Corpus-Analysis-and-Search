# Corpus Engine: A Web-Based Text Analysis and Search Tool

Corpus Engine lets you upload, analyze, and search text files with ease. It offers real-time insights like token statistics, lexical diversity, and similarity-based search results in a modern, responsive web interface.

## Key Features
- **Upload Files**: Drag and drop `.txt` files or select them manually.
- **Analyze Corpus**: View detailed metrics, including total tokens, unique tokens, lexical diversity, and formality ratio.
- **Search**: Query the corpus for the most relevant files using TF-IDF similarity.
- **Reload**: Refresh the corpus to include newly uploaded files.
- **Real-Time Feedback**: See upload status, analysis, and search results instantly.

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/corpus-engine.git
   cd corpus-engine
   ```

2. Run the environment setup script:
   ```bash
   ./setup_env.sh
   ```
   This will create a Python virtual environment and install all required dependencies.

3. Start the application:
   ```bash
   ./app_run.sh
   ```

4. Open your browser at:
   ```
   http://127.0.0.1:5000/
   ```

## Folder Structure
```
project/
├── app.py          # Flask app
├── corpus.py       # Corpus logic
├── setup_env.sh    # Environment setup script
├── app_run.sh      # Script to start the app
├── templates/      # HTML files
├── static/         # CSS/JS files
├── texts/          # Uploaded files
├── frequencies/    # Processed data
```

## Usage
1. **Upload Files**: Drag and drop `.txt` files or use the "Select Files" button.
2. **Analyze**: Click "Analyze" to get metrics for all files in the corpus.
3. **Search**: Enter a query to find similar files using TF-IDF similarity.
4. **Reload**: Reload the corpus after uploading new files.

## License
Licensed under the MIT License.
