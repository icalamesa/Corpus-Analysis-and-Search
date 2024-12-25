# Corpus Analysis and Query System

## Overview
This Python code provides a framework for processing, analyzing, and querying a collection of text files using Natural Language Processing (NLP) techniques. It uses TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity to evaluate text similarities and perform efficient searches.

## Key Functionalities

### 1. **Corpus Initialization**
- The class initializes with:
  - `text_folder`: The directory containing text files to be analyzed.
  - `frequencies_dir`: The directory to save output files.
- Automatically processes all text files in the specified directory during initialization.
- Handles missing files with warnings.

### 2. **Text Preprocessing**
- Tokenizes text into words.
- Lemmatizes tokens to their base form.
- Removes stopwords and non-alphabetic tokens.
- Computes part-of-speech (POS) tags for all tokens.

### 3. **Metrics Calculation**
The following metrics are computed for each text file:
- **Total Tokens**: The number of tokens in the text.
- **Unique Tokens**: The count of distinct tokens.
- **Lexical Diversity**: Ratio of unique tokens to total tokens.
- **Average Token Length**: The mean length of tokens.
- **Formality Ratio**: The proportion of long words (>6 characters).
- **POS Distribution**: Frequency of part-of-speech tags (e.g., nouns, verbs).

### 4. **TF-IDF Calculation**
- Calculates TF-IDF scores for the entire corpus.
- Stores the results for each document as a DataFrame.
- Prepares the vectorizer for future queries.

### 5. **Query Processing**
- Allows users to input a query and compute its:
  - **TF-IDF Scores**: Ranks terms in the query based on importance.
  - **Cosine Similarity**: Measures the similarity of the query to each document in the corpus.
- Outputs a ranked list of documents by relevance.

### 6. **Data Saving**
- Saves processed dataframes (e.g., token frequencies, TF-IDF scores) to disk in CSV or JSON format.

### 7. **Visualization**
- Generates bar charts for token frequency distributions across documents.

## How to Use

### 1. **Initialize the Corpus**
```python
corpus = Corpus(text_folder='./texts', frequencies_dir='./frequencies')
```

### 2. **Analyze Texts**
To retrieve metrics and insights for all processed texts:
```python
print(corpus.analyze_all_texts())
```

### 3. **Query the Corpus**
To compute TF-IDF and cosine similarity for a search query:
```python
query = "country and politics"
similarity_df = corpus.compute_query_similarity(query)
print(similarity_df)
```

### 4. **Visualize Frequency Distributions**
To plot the top token frequencies for each text:
```python
corpus.plot_graph()
```

## Dependencies
- **Python Libraries**:
  - `pandas`
  - `matplotlib`
  - `nltk`
  - `scikit-learn`

- **NLTK Resources** (downloaded automatically):
  - `punkt`
  - `averaged_perceptron_tagger`
  - `stopwords`
  - `wordnet`

## Example Output

### Query Similarity Output
For a query like "machine learning and data analysis":
```plaintext
                      File Path  Similarity
0         ./texts/document1.txt    0.732109
1         ./texts/document3.txt    0.654321
2         ./texts/document2.txt    0.523456
3         ./texts/document4.txt    0.432198
```

### Token Frequency Visualization
Generates bar charts showing the top 20 tokens by frequency for each document.

---

## Summary
This framework is designed for efficient text processing and query analysis, making it ideal for tasks like document similarity evaluation, keyword extraction, and search applications.
