import os
import re

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(ROOT, "models", "regressors")
DIFF_PATH = os.path.join(MODELS_DIR, "difficulty_regressor.pkl")
IMP_PATH = os.path.join(MODELS_DIR, "importance_regressor.pkl")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
SVD_PATH = os.path.join(MODELS_DIR, "tfidf_svd.pkl")

URGENT = {'urgent', 'asap', 'critical', 'deadline', 'important', 'immediately'}
HARD = {'hard', 'difficult', 'complex', 'tough', 'challenging', 'heavy', 'intense'}
EASY = {'easy', 'simple', 'quick', 'light', 'trivial', 'basic', 'gentle'}
TIME = {'minute', 'hour', 'day', 'week', 'month', 'today', 'tomorrow'}
NEGATION = {'not', 'no', 'never', 'without', "n't"}
INTENSIFIERS = {'very', 'extremely', 'super', 'incredibly', 'insanely', 'really', 'highly', 'especially', 'particularly'}
HEDGES = {'kind of', 'sort of', 'somewhat', 'fairly', 'rather', 'pretty', 'quite', 'slightly', 'barely'}


def _is_negated(words, i):
    for j in range(max(0, i - 2), i):
        if words[j] in NEGATION or words[j].endswith("n't"):
            return True
    return False


def extract_lexical_features(texts):
    feats = []
    for t in texts:
        w = t.lower().split()
        n_chars = len(t)
        n_words = len(w)

        urgent_raw = hard_raw = easy_raw = time_raw = 0
        urgent_neg = hard_neg = easy_neg = time_neg = 0
        negation_count = intense_count = 0

        for i, x in enumerate(w):
            if x in NEGATION or x.endswith("n't"):
                negation_count += 1
            if x in INTENSIFIERS:
                intense_count += 1
            negated = _is_negated(w, i)
            if x in URGENT:
                if negated:
                    urgent_neg += 1
                else:
                    urgent_raw += 1
            if x in HARD:
                if negated:
                    hard_neg += 1
                else:
                    hard_raw += 1
            if x in EASY:
                if negated:
                    easy_neg += 1
                else:
                    easy_raw += 1
            if x in TIME:
                if negated:
                    time_neg += 1
                else:
                    time_raw += 1

        full_lower = t.lower()
        hedge_count = sum(1 for h in HEDGES if h in full_lower)

        feats.append([
            n_words, n_chars, n_chars / max(n_words, 1),
            t.count('!'),
            sum(1 for c in t if c.isupper()),
            sum(1 for c in t if c.isupper()) / max(n_chars, 1),
            urgent_raw, hard_raw, easy_raw, time_raw,
            urgent_neg, hard_neg, easy_neg, time_neg,
            negation_count, intense_count, hedge_count,
            int(bool(re.search(r'\d+', t))),
        ])
    return np.array(feats)


class RegressorPredictor:
    def __init__(self):
        self.encoder = SentenceTransformer('all-mpnet-base-v2')
        self.diff_model = joblib.load(DIFF_PATH)
        self.imp_model = joblib.load(IMP_PATH)
        self.tfidf = joblib.load(TFIDF_PATH)
        self.svd = joblib.load(SVD_PATH)

    def _build_features(self, text):
        emb = self.encoder.encode(text, show_progress_bar=False)
        feats = extract_lexical_features(text)
        tfidf_raw = self.tfidf.transform(text)
        tfidf_svd = self.svd.transform(tfidf_raw)
        return np.concatenate([emb, feats, tfidf_svd], axis=1)

    def predict(self, text: str):
        if isinstance(text, str):
            text = [text]
        X = self._build_features(text)
        d = float(np.clip(self.diff_model.predict(X), 0, 1)[0])
        i = float(np.clip(self.imp_model.predict(X), 0, 1)[0])
        return d, i
