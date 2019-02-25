import torch
from ..base import BaseEstimator
from ..base import RegressorMixin
from ..utils import set_device
from ..utils import pseudo_inverse, preprocess_data

#
# def preprocess_data(X, y, fit_intercept, normalize=False, copy=True):
#     if copy:
#         X = X.clone()
#         y = y.clone()
#     if fit_intercept:
#         X_offset = X.mean(dim=0)
#         y_offset = y.mean(dim=0)
#         X.sub_(X_offset)
#         y.sub_(y_offset)
#         if normalize:
#             X_scale = X.norm(2, dim=0)
#             X.div_(X_scale)
#         else:
#             X_scale = torch.ones_like(X[0])
#     else:
#         X_offset = torch.zeros_like(X[0])
#         y_offset = torch.zeros_like(y[0])
#         X_scale = torch.ones_like(X[0])
#     return X, y, X_offset, y_offset, X_scale


class LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=0, fit_intercept=True, normalize=False, copy_X=True, device='cuda', verbose=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.device = set_device(device, verbose=verbose)

    def fit(self, X, y):
        X = X.to(self.device)
        y = y.to(device=self.device, dtype=X.dtype)
        X, y, X_offset, y_offset, X_scale = preprocess_data(X, y,
                                                            fit_intercept=self.fit_intercept,
                                                            normalize=self.normalize,
                                                            copy=self.copy_X)
        betas = pseudo_inverse(X, alpha=self.alpha) @ y
        self.coef_ = betas.view(-1, ) / X_scale
        self.intercept_ = y_offset - (X_offset / X_scale) @ betas
        return self

    def predict(self, X):
        X = X.to(self.device)
        y_pred = self.intercept_ + X @ self.coef_.view(-1, 1)
        return y_pred

    def score(self, X, y, score_func=None, **kwargs):
        X = X.to(self.device)
        y = y.to(device=self.device, dtype=X.dtype)
        return super(LinearRegression, self).score(X, y, score_func=score_func, **kwargs)


class Lasso(BaseEstimator, RegressorMixin):
    pass
