import torch


def pseudo_inverse(X, alpha=0):
    """Calculates the pseudo-inverse of a 2D tensor using SVD.

    :param X: input matrix
    :param alpha: L2 regularization strength; must be a positive float.
    :return: pseudo-inverse of X with l2 regularization
    """
    """

    :param X: input matrix
    :return: pseudo-inverse of X
    """
    u, s, v = X.svd(some=True)
    s_inv = s / (s.pow(2) + alpha)
    return v @ s_inv.diag() @ u.t()


def preprocess_data(X, y, fit_intercept, normalize=False, copy=True):
    if copy:
        X = X.clone()
        y = y.clone()
    if fit_intercept:
        X_offset = X.mean(dim=0)
        y_offset = y.mean(dim=0)
        X.sub_(X_offset)
        y.sub_(y_offset)
        if normalize:
            X_scale = X.norm(2, dim=0)
            X.div_(X_scale)
        else:
            X_scale = torch.ones_like(X[0])
    else:
        X_offset = torch.zeros_like(X[0])
        y_offset = torch.zeros_like(y[0])
        X_scale = torch.ones_like(X[0])
    return X, y, X_offset, y_offset, X_scale
