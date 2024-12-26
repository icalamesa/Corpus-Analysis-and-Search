#!/bin/bash

if [ ! -d "env" ]; then
  echo "Environment not found. Running setup_env.sh..."
  ./setup_env.sh
fi
source env/bin/activate

echo "Starting the Corpus Engine app..."
python3 app.py
