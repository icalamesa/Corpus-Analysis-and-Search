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
        self._process_corpus()

    def _process_corpus(self):
        """
        Process all text files in the directory: tokenize, preprocess, analyze, and compute TF-IDF.
        """
        raw_texts = []

        for file_path in self.file_paths:
            print(f"Processing {file_path}...")
            text = self._read_text_from_file(file_path)
            text_df = self._tokenize_and_preprocess(text)
            metrics = self._calculate_metrics(text_df)
            token_data_df = self._generate_token_data_dataframe(text_df)
            self.data_model[file_path] = {
                'text_df': text_df,
                'token_data_df': token_data_df,
                'metrics': metrics
            }
            
            raw_texts.append(" ".join(text_df['token'].tolist()))

        self._compute_tfidf(raw_texts)
        self._save_all_dataframes()


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
            'File Path': list(self.data_model.keys()),
            'Similarity': similarities
        }).sort_values(by='Similarity', ascending=False)
        
        return results



    # Private methods

    def _read_text_from_file(self, filepath: str) -> str:
        with open(filepath, "r") as file:
            return file.read().replace('\n', '')

    def _get_files_from_directory(self, path: str) -> list:
        return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.txt')]

    def _tokenize_and_preprocess(self, text: str) -> pd.DataFrame:
        tokens = word_tokenize(text.lower())
        text_df = pd.DataFrame(tokens, columns=['token'])
        return self._preprocess_text_df(text_df)

    def _preprocess_text_df(self, text_df: pd.DataFrame) -> pd.DataFrame:
        text_df['token'] = text_df['token'].apply(self.lemmatizer.lemmatize)
        pos_tags = pos_tag(text_df['token'].tolist())
        text_df['pos_tag'] = [tag for _, tag in pos_tags]

        text_df = text_df[text_df['token'].str.isalpha()]
        stop_words = set(stopwords.words('english'))
        text_df = text_df[~text_df['token'].isin(stop_words)]
        text_df = text_df[text_df['token'].apply(lambda x: len(x) >= 3)]
        return text_df

    def _generate_token_data_dataframe(self, text_df: pd.DataFrame) -> pd.DataFrame:
        frequency_series = text_df['token'].value_counts()
        token_data_df = frequency_series.reset_index()
        token_data_df.columns = ['Token', 'Count']
        min_count = token_data_df['Count'].min()
        max_count = token_data_df['Count'].max()
        token_data_df['minmax_norm'] = (token_data_df['Count'] - min_count) / (max_count - min_count)
        token_data_df['Index'] = token_data_df.index
        return token_data_df

    def _compute_tfidf(self, raw_texts: list):
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

if __name__ == "__main__":
    corpus = Corpus()
    #corpus.plot_graph()
    print(corpus.analyze_all_texts())
    print(corpus.search("America is great again"))