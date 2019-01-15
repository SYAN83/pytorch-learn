

def pseudo_inverse(X):
    """Calculates the pseudo-inverse of a 2D tensor using SVD.

    :param X: input matrix
    :return: pseudo-inverse of X
    """
    u, s, v = X.svd(some=True)
    return v @ s.pow(-1).diag() @ u.t()
