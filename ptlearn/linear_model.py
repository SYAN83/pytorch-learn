import torch
from .base import BaseEstimator, RegressorMixin
from .preprocessing import Normalizer
from .utils import set_device


def make_dataset(X, y=None, fit_intercept=False, normalizer=None, fit_normalizer=False, device=None):
    if device is None:
        device = X.device
    else:
        X = X.to(device)
    if fit_intercept:
        if normalizer is not None:
            if fit_normalizer:
                X = normalizer.fit_transform(X)
            else:
                X = normalizer.transform(X)
        X = torch.cat((X, torch.ones((X.size(0), 1), device=device)), 1)
    if y is None:
        return X
    else:
        assert X.size(0) == y.size(0), 'X and y do not have the same size.'
        y = y.to(X.dtype).to(X.device)
        if y.dim() == 1:
            y = y.view(-1, 1)
        return X, y


class LinearRegression(BaseEstimator, RegressorMixin):
    
    def __init__(self, fit_intercept=True, normalize=False, device='cuda', verbose=True):
        self.fit_intercept = fit_intercept
        self.normalizer = Normalizer() if normalize else None
        self.verbose = verbose
        self.device = set_device(device=device, verbose=self.verbose)
        self._betas = None
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        x, y = make_dataset(X, y, fit_intercept=self.fit_intercept,
                            normalizer=self.normalizer, fit_normalizer=True,
                            device=self.device)
        # compute betas using normal equation
        # self._betas = (x.t() @ x).inverse() @ x.t() @ y
        # compute betas using pseudo inverse of X
        self._betas = X.pinverse() @ y
        self.coef_ = self._betas.cpu().numpy()[:X.size(0)]
        if self.fit_intercept:
            self.intercept_ = self._betas.cpu().numpy()[-1]
        else:
            self.intercept_ = None
        return self
    
    def predict(self, X):
        x = make_dataset(X, fit_intercept=self.fit_intercept,
                         normalizer=self.normalizer, fit_normalizer=False,
                         device=self.device)
        y_pred = x @ self._betas
        return y_pred


class Lasso(LinearRegression):

    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
