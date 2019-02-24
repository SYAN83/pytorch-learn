
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
