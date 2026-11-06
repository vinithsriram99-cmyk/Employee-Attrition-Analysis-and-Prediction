"""Custom scikit-learn steps used by the model pipeline."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class IQRClipper(BaseEstimator, TransformerMixin):
    """Outlier handling: clip each numeric column to [Q1 - 1.5*IQR, Q3 + 1.5*IQR] learned from the training data.
    Columns with no spread (IQR = 0) are left unchanged."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(values, [25, 75], axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        flat = iqr == 0
        self.lower_[flat], self.upper_[flat] = -np.inf, np.inf
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
