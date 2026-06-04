from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TextEntropyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        entropies = []

        for value in X:
            text = str(value)
            if not text:
                entropies.append(0.0)
                continue

            length = len(text)
            counts = Counter(text)
            entropy = -sum(
                (count / length) * np.log2(count / length) for count in counts.values()
            )
            entropies.append(entropy)

        return np.array(entropies, dtype=float).reshape(-1, 1)
