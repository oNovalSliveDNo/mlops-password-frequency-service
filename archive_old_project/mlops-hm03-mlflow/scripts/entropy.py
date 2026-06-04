from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from collections import Counter
import math


# Custom transformer for entropy calculation
class TextEntropyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        entropies = []
        for text in X:
            # Calculate entropy for each text
            freq = Counter(text)
            prob = [f / len(text) for f in freq.values()]
            entropy = -sum(p * math.log2(p) for p in prob)
            entropies.append([entropy])
        return np.array(entropies)
