#!/bin/bash

# Exit on error
set -e

# Environment name
ENV_NAME="corpus_env"

# Create Python virtual environment
echo "Creating virtual environment: $ENV_NAME..."
python3 -m venv $ENV_NAME

# Activate the environment
echo "Activating virtual environment..."
source $ENV_NAME/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install necessary libraries
echo "Installing required libraries..."
pip install \
    pandas \
    matplotlib \
    nltk \
    scikit-learn \
    fastapi \
    uvicorn

# Download NLTK datasets
echo "Downloading NLTK datasets..."
python -m nltk.downloader punkt averaged_perceptron_tagger stopwords wordnet

# Display success message
echo "Environment setup complete! To activate, run: source $ENV_NAME/bin/activate"
