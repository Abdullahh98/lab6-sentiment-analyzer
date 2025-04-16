import os
os.environ["THINC_BACKEND"] = "cpu"  # 🚫 Disable GPU usage

import numpy as np
import pandas as pd
import regex as re
import joblib
import spacy  # ✅ Use spacy directly (instead of en_core_web_sm)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC

# ✅ Load the small English model with minimal pipeline to avoid issues
nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner"])
classifier = LinearSVC()

def clean_text(text):
    # reduce multiple spaces and newlines to only one
    text = re.sub(r'(\s\s+|\n\n+)', r'\1', text)
    # remove double quotes
    text = re.sub(r'"', '', text)

    return text

def convert_text(text):
    sent = nlp(text)
    tokens = []
    for w in sent:
        if w.is_stop or w.is_punct:
            continue
        tokens.append(w.lemma_.lower())
    text = ' '.join(tokens)

    return text

class preprocessor(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(clean_text).apply(convert_text)
