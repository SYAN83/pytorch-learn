from .base import BaseEstimator


__all__ = [
    'Normalizer'
]


class Normalizer(BaseEstimator):

    def __init__(self, mean=None, std=None, eps=1e-05):
        self.mean_ = mean
        self.std_ = std
        self.eps = eps

    def fit(self, X):
        if self.mean_ is None:
            self.mean_ = X.mean(dim=0)
        if self.std_ is None:
            self.std_ = (X.var(dim=0) - self.eps).sqrt()
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
