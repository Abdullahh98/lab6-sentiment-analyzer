import numpy as np
import pandas as pd
import regex as re
import spacy
from sklearn.base import BaseEstimator, TransformerMixin

# Load spaCy model globally
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = re.sub(r'(\s\s+|\n\n+)', r'\1', text)
    text = re.sub(r'"', '', text)
    return text

def convert_text(text):
    sent = nlp(text)
    ents = {x.text: x for x in sent.ents}
    tokens = []
    for w in sent:
        if w.is_stop or w.is_punct:
            continue
        if w.text in ents:
            tokens.append(w.text)
        else:
            tokens.append(w.lemma_.lower())
    return ' '.join(tokens)

class preprocessor(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(clean_text).apply(convert_text)
