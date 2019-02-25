import torch
from .base import SGDBaseEstimator
from ..base import RegressorMixin


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, in_features):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(in_features=in_features,
                                      out_features=1,
                                      bias=False)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

    @property
    def coef_(self):
        return self.linear.weight.data

    @property
    def intercept_(self):
        return self.linear.bias.data


class LinearRegression(SGDBaseEstimator, RegressorMixin):

    def fit(self, X, y):
        self.model = LinearRegressionModel(in_features=X.shape[1]).to(device=self.device, dtype=X.dtype)
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, weight_decay=self.alpha)
        self._fit(X=X, y=y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        return self


