# N-Gram Language Model

A statistical language model implemented in **Python** using **NLTK**.  
Supports **unigram** and **bigram** models with Add-*k* smoothing, sentence probability calculation, and perplexity scoring.

## Features
- **Custom Implementation** of unigram and bigram models from scratch.
- **Text Preprocessing**: tokenization, punctuation removal, and train/test splitting.
- **Add-*k* Smoothing** to handle unseen n-grams and improve generalization.
- **Sentence Probability Calculation** for model evaluation.
- **Perplexity Scoring** to compare model performance.

## Installation
```bash
# Clone this repository
git clone https://github.com/<your-username>/ngram_language_model.git
cd ngram_language_model

# (Optional) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
