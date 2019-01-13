import torch
from .base import BaseEstimator
from .metrics import r2_score
from .utils import set_device


def make_dataset(X, y=None, fit_intercept=False, device=torch.device('cpu')):
    x = X.to(device)
    if fit_intercept:
        x = torch.cat((x, torch.ones((X.size(0), 1), device=device)), 1)
    if y is not None:
        y = y.to(x.dtype).to(x.device)
    return x, y


class LinearRegression(BaseEstimator):
    
    def __init__(self, fit_intercept=True, normalize=False, device='cuda', verbose=True):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.verbose = verbose
        self.device = set_device(device=device, verbose=self.verbose)
        self._betas = None
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X, y):
        x, y = make_dataset(X, y, device=self.device)
        
        self._betas = (x.t() @ x).inverse() @ x.t() @ y
        self.coef_ = self._betas.cpu().numpy()[:X.size(0)]
        if self.fit_intercept:
            self.intercept_ = self._betas.cpu().numpy()[-1]
    
    def predict(self, X):
        x, _ = make_dataset(X, device=self.device)
        if self.fit_intercept:
            x = torch.cat((x, torch.ones((X.size(0), 1), device=self.device)), 1)
        y_pred = x @ self._betas
        return y_pred
    
    def score(self, X, y):
        y_pred = self.predict(X)
        score_ = r2_score(y_true=y, y_pred=y_pred)
        return score_
