
__all__ = [
    'LinearRegression', 'Ridge'
]

from abc import abstractmethod
import torch
from .base import BaseEstimator, RegressorMixin
from .utils import set_device
from .utils.tensor import pseudo_inverse


def _transform(x, fit_intercept=False, normalize=False, device=None, eps=1e-5):
    """

    :param X: Training data / Target values
    :param fit_intercept: Whether to calculate the intercept for this model.
    :param normalize: This parameter is ignored when ``fit_intercept`` is set to False.
    :param device:
    :param eps: term added to the denominator to improve numerical stability
    :return:
    """
    if device:
        x = x.to(device)
    mean_, std_ = torch.zeros_like(x[0]), torch.ones_like(x[0])
    if fit_intercept:
        mean_ = x.mean(dim=0)
        x = x - mean_
        if normalize:
            std_ = (x.var(dim=0) - eps).sqrt()
            x = x / std_
    return x, mean_, std_


class _BaseLinearRegression(BaseEstimator):

    @abstractmethod
    def __init__(self, fit_intercept, normalize, device, verbose):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.verbose = verbose
        self.device = set_device(device=device, verbose=self.verbose)
        super(BaseEstimator, self).__init__()

    @abstractmethod
    def _fit(self, X, y, alpha=0.0):
        assert X.size(0) == y.size(0), 'X and y do not have the same size.'
        X_ = X.to(self.device)
        X__, x_mean, x_std = _transform(x=X_,
                                        fit_intercept=self.fit_intercept,
                                        normalize=self.normalize)
        y_ = y.to(self.device)
        if y_.dim() == 1:
            y_ = y_.view(-1, 1)
        y__, y_mean, y_std = _transform(x=y_,
                                        fit_intercept=self.fit_intercept,
                                        normalize=self.normalize)
        betas = pseudo_inverse(X__, alpha=alpha) @ y__
        self.coef_ = y_std * (betas.view(-1,)/x_std)
        self.intercept_ = y_mean - y_std * (x_mean/x_std) @ betas
        return self

    def predict(self, X):
        X_ = X.to(self.device)
        y_pred = self.intercept_ + X_ @ self.coef_.view(-1,1)
        return y_pred


class LinearRegression(_BaseLinearRegression, RegressorMixin):
    
    def __init__(self, fit_intercept=True, normalize=False, device='cuda', verbose=True):
        self.coef_ = None
        self.intercept_ = None
        super(LinearRegression, self).__init__(fit_intercept=fit_intercept,
                                               normalize=normalize,
                                               device=device,
                                               verbose=verbose)

    def fit(self, X, y):
        return super(LinearRegression, self)._fit(X, y)


class Ridge(_BaseLinearRegression, RegressorMixin):

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, device='cuda', verbose=True):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        super(Ridge, self).__init__(fit_intercept=fit_intercept,
                                    normalize=normalize,
                                    device=device,
                                    verbose=verbose)

    def fit(self, X, y):
        return super(Ridge, self)._fit(X, y, alpha=self.alpha)