import os
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.metrics.pairwise import cosine_similarity


# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

class Corpus:
    def __init__(self, text_folder='./texts/', frequencies_dir='./frequencies'):
        self.text_folder = Path(text_folder).resolve()
        self.frequencies_dir = Path(frequencies_dir).resolve()
        self.lemmatizer = WordNetLemmatizer()
        self.file_paths = self._get_files_from_directory(self.text_folder)
        self.vectorizer = TfidfVectorizer()

        self.data_model = {}  # Core data model for storing processed data

        # Ensure the directories exist
        os.makedirs(self.frequencies_dir, exist_ok=True)

        # Warn if no files found
        if not self.file_paths:
            print(f"Warning: No text files found in {self.text_folder}.")

        # Process all text data upon initialization
        self.update_corpus()

    def list_files(self) -> list:
        return list(self.data_model.keys())

    def update_corpus(self, file_paths_list: list = None) -> dict:
        """
        Process all text files in the directory: tokenize, preprocess, analyze, and compute TF-IDF.
        
        Args:
            file_paths_list (list): List of file paths to process. If None, processes all files in self.file_paths.
        
        Returns:
            dict: Summary of processing, including successfully added files and skipped files with reasons.
        """
        raw_texts = []
        skipped_files = []  # Collect details about skipped files
        added_files = []  # Collect successfully added files

        if file_paths_list is None:
            file_paths_list = self.file_paths

        for file_path in file_paths_list:
            if not file_path.endswith(".txt"):
                skipped_files.append({
                    "file": file_path,
                    "reason": "Invalid file format. Only .txt files are supported."
                })
                continue

            try:
                print(f"Processing {file_path}...")
                text = self._read_text_from_file(file_path)
                text_df = self._tokenize_and_preprocess(text)
                metrics = self._calculate_metrics(text_df)
                token_data_df = self._generate_token_data_dataframe(text_df)

                # Update the data model with processed data
                self.data_model[file_path] = {
                    'text_df': text_df,
                    'token_data_df': token_data_df,
                    'metrics': metrics
                }
                raw_texts.append(" ".join(text_df['token'].tolist()))
                added_files.append(file_path)  # Add to successful files

            except Exception as e:
                skipped_files.append({
                    "file": file_path,
                    "reason": f"Error during processing: {str(e)}"
                })
                continue

        # Compute TF-IDF for valid files
        if raw_texts:
            self._compute_tfidf(raw_texts)
            self._save_all_dataframes()

        # Return a summary of added and skipped files
        return {
            "added_files": added_files,
            "skipped_files": skipped_files
        }


    def _save_dataframe(self, df: pd.DataFrame, filename: str, file_format: str = 'csv') -> None:
        """
        Save a single dataframe to disk in the specified format.
        """
        target_path = os.path.join(self.frequencies_dir, f"{filename}.{file_format}")

        try:
            if file_format == 'csv':
                df.to_csv(target_path, index=False)
                print(f"DataFrame saved as CSV at {target_path}")
            elif file_format == 'json':
                with open(target_path, 'w') as json_file:
                    json.dump(df.to_dict(orient='records'), json_file, indent=4)
                print(f"DataFrame saved as JSON at {target_path}")
            else:
                raise ValueError(f"Unsupported file format: {file_format}. Please use 'csv' or 'json'.")
        except Exception as e:
            print(f"Failed to save dataframe {filename}: {e}")

    def _save_all_dataframes(self):
        """
        Save all processed dataframes to disk after processing is complete.
        """
        for file_path, data in self.data_model.items():
            filename = os.path.splitext(os.path.basename(file_path))[0]
            if 'token_data_df' in data:
                self._save_dataframe(data['token_data_df'], f"{filename}_frequencies")
            if 'tfidf_df' in data:
                self._save_dataframe(data['tfidf_df'], f"{filename}_tfidf")

    def analyze_all_texts(self) -> str:
        """
        Retrieve and format insights for all processed texts in a user-friendly way.
        """
        #analyses = [self._analyze_text(file_path) for file_path in self.data_model]
        #return json.dumps(analyses, indent=4)
        df_list = []
        for key, text in self.data_model.items():
            df_list.append(pd.DataFrame({os.path.basename(key) : text['metrics']}))
        return pd.concat(df_list, axis=1)

    # Private methods
    def _analyze_text(self, file_path: str) -> dict:
        """
        Retrieve precomputed metrics and insights for a specific text file.
        """
        if file_path in self.data_model:
            data = self.data_model[file_path]
            return {
                "File Path": file_path,
                "Metrics": {
                    "Total Tokens": data['metrics']['total_tokens'],
                    "Unique Tokens": data['metrics']['unique_tokens'],
                    "Lexical Diversity": data['metrics']['lexical_diversity'],
                    "Average Token Length": data['metrics']['average_token_length'],
                    "Formality Ratio": data['metrics']['formality_ratio'],
                },
                "POS Distribution": data['metrics']['pos_distribution'],
                "Top 10 TF-IDF Terms": data.get('tfidf_df', pd.DataFrame()).head(10).to_dict(orient='records')
            }
        else:
            raise ValueError(f"File {file_path} not processed.")


    def _calculate_metrics(self, text_df: pd.DataFrame) -> dict:
        """
        Calculate various metrics for the given text data.
        """
        total_tokens = len(text_df)
        unique_tokens = text_df['token'].nunique()
        lexical_diversity = unique_tokens / total_tokens if total_tokens > 0 else 0
        average_token_length = text_df['token'].str.len().mean() if total_tokens > 0 else 0
        pos_distribution = text_df['pos_tag'].value_counts().to_dict()
        formality_ratio = self._analyze_formality(text_df)

        return {
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "lexical_diversity": lexical_diversity,
            "average_token_length": average_token_length,
            "pos_distribution": pos_distribution,
            "formality_ratio": formality_ratio
        }

    def plot_graph(self):
        """
        Plot frequency graphs for all processed text files.
        """
        num_files = len(self.data_model)
        rows = (num_files // 2) + (num_files % 2)
        cols = 2 if num_files > 1 else 1

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), sharey=True)
        axes = axes.flatten() if num_files > 1 else [axes]

        for ax, (file_path, data) in zip(axes, self.data_model.items()):
            token_data_df = data['token_data_df']
            top_words = token_data_df.head(20)
            ax.bar(top_words['Token'], top_words['minmax_norm'], color='grey')
            ax.set_xticklabels(top_words['Token'], rotation=90)
            ax.set_xlabel('Words')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Top Words - {os.path.basename(file_path)}')

        plt.tight_layout()
        plt.show()

    def search(self, query: str) -> pd.DataFrame:
        """
        Compute the cosine similarity of the query against all documents in the corpus.

        Args:
            query (str): The search query string.

        Returns:
            pd.DataFrame: A DataFrame containing file paths and their similarity scores.
        """
        if not hasattr(self, 'vectorizer'):
            raise ValueError("TF-IDF vectorizer not initialized. Ensure _compute_tfidf has been run.")
        
        # Transform the query into the same vector space
        query_vector = self.vectorizer.transform([query])
        
        # Compute cosine similarity
        tfidf_matrix = self.vectorizer.transform([" ".join(data['text_df']['token'].tolist()) for data in self.data_model.values()])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Create a DataFrame with results
        results = pd.DataFrame({
            'File': list([os.path.basename(key) for key in self.data_model.keys()]),
            'Similarity': [round(similarity,2) for similarity in similarities]
        }).sort_values(by='Similarity', ascending=False)
        
        return results



    # Private methods

    def _read_text_from_file(self, filepath: str) -> str:
        """
        Reads the content of a text file and returns it as a single string.

        Args:
            filepath (str): Path to the text file.

        Returns:
            str: The content of the file as a single string with newline characters removed.

        Scope:
            - Reads raw text from an external file.
            - Does not modify or interact with the `Corpus` data model.
        """
        with open(filepath, "r") as file:
            return file.read().replace('\n', '')

    def _get_files_from_directory(self, path: str) -> list:
        """
        Retrieves a list of `.txt` files from a specified directory.

        Args:
            path (str): Path to the directory to search for text files.

        Returns:
            list: A list of file paths for all `.txt` files in the directory.

        Scope:
            - Reads directory contents.
            - Does not modify or interact with the `Corpus` data model.
        """
        return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]

    def _tokenize_and_preprocess(self, text: str) -> pd.DataFrame:
        """
        Tokenizes and preprocesses a raw text string into a structured DataFrame.

        Args:
            text (str): Raw text to be tokenized and preprocessed.

        Returns:
            pd.DataFrame: A DataFrame with a single column (`token`) containing processed tokens.

        Scope:
            - Reads raw text (input).
            - Outputs a structured DataFrame with basic tokenization and normalization.
            - Does not directly interact with the `Corpus` data model.
        """
        tokens = word_tokenize(text.lower())
        text_df = pd.DataFrame(tokens, columns=['token'])
        return self._preprocess_text_df(text_df)

    def _preprocess_text_df(self, text_df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs advanced preprocessing on a DataFrame of tokens:
        - Lemmatization
        - Part-of-speech tagging
        - Removal of non-alphabetic tokens
        - Removal of stopwords
        - Filtering of short tokens (<3 characters)

        Args:
            text_df (pd.DataFrame): A DataFrame with a `token` column to preprocess.

        Returns:
            pd.DataFrame: The preprocessed DataFrame with additional columns like `pos_tag`.

        Scope:
            - Reads and modifies a `text_df`.
            - Adds part-of-speech tagging and filters tokens.
            - Does not directly interact with the `Corpus` data model.
        """
        text_df['token'] = text_df['token'].apply(self.lemmatizer.lemmatize)
        pos_tags = pos_tag(text_df['token'].tolist())
        text_df['pos_tag'] = [tag for _, tag in pos_tags]

        text_df = text_df[text_df['token'].str.isalpha()]
        stop_words = set(stopwords.words('english'))
        text_df = text_df[~text_df['token'].isin(stop_words)]
        text_df = text_df[text_df['token'].apply(lambda x: len(x) >= 3)]
        return text_df

    def _generate_token_data_dataframe(self, text_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a DataFrame with token frequency counts and normalized scores.

        Args:
            text_df (pd.DataFrame): A DataFrame with a `token` column to analyze.

        Returns:
            pd.DataFrame: A DataFrame with the following columns:
                - `Token`: Unique tokens.
                - `Count`: Frequency of each token.
                - `minmax_norm`: Min-max normalized frequencies.
                - `Index`: Token indices.

        Scope:
            - Reads a `text_df` for frequency analysis.
            - Outputs a token analysis DataFrame.
            - Does not directly interact with the `Corpus` data model.
        """
        frequency_series = text_df['token'].value_counts()
        token_data_df = frequency_series.reset_index()
        token_data_df.columns = ['Token', 'Count']
        min_count = token_data_df['Count'].min()
        max_count = token_data_df['Count'].max()
        token_data_df['minmax_norm'] = (token_data_df['Count'] - min_count) / (max_count - min_count)
        token_data_df['Index'] = token_data_df.index
        return token_data_df

    def _compute_tfidf(self, raw_texts: list):
        """
        Computes TF-IDF scores for a collection of raw texts and updates the data model.

        Args:
            raw_texts (list): A list of raw text strings, each representing a document in the corpus.

        Returns:
            None: Updates the `self.data_model` with the following for each file:
                - `tfidf_df`: A DataFrame containing tokens and their corresponding TF-IDF scores.

        Scope:
            - Reads raw text (input).
            - Modifies the `self.data_model` by adding a `tfidf_df` for each file.
            - Uses the vectorizer to compute TF-IDF for all texts.
        """
        tfidf_matrix = self.vectorizer.fit_transform(raw_texts)
        feature_names = self.vectorizer.get_feature_names_out()

        for idx, file_path in enumerate(self.file_paths):
            tfidf_scores = tfidf_matrix[idx].toarray().flatten()
            tokens = feature_names
            tfidf_df = pd.DataFrame({
                'Token': tokens,
                'TFIDF': tfidf_scores
            }).sort_values(by='TFIDF', ascending=False)
            self.data_model[file_path]['tfidf_df'] = tfidf_df


    def _compute_query_tfidf(self, query: str) -> pd.DataFrame:
        """
        Compute the TF-IDF scores for a given query against the corpus.

        Args:
            query (str): The search query string.

        Returns:
            pd.DataFrame: A DataFrame containing tokens and their TF-IDF scores for the query.
        """
        if not hasattr(self, 'vectorizer'):
            raise ValueError("TF-IDF vectorizer not initialized. Ensure _compute_tfidf has been run.")
        
        query_vector = self.vectorizer.transform([query])
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_scores = query_vector.toarray().flatten()
        query_tfidf_df = pd.DataFrame({
            'Token': feature_names,
            'TFIDF': tfidf_scores
        }).sort_values(by='TFIDF', ascending=False)
        
        return query_tfidf_df


    def _analyze_formality(self, text_df: pd.DataFrame) -> float:
        """
        Calculate the formality ratio based on token length.
        """
        long_words = text_df[text_df['token'].str.len() > 6]
        return len(long_words) / len(text_df) if not text_df.empty else 0
