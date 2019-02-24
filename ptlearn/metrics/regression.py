import torch
from torch.nn.functional import l1_loss, mse_loss
from functools import partial


def scoring(y_true, y_pred, score_func, **kwargs):
    with torch.no_grad():
        y_true = y_true.to(y_pred.device).view(y_pred.shape)
        score = score_func(y_true, y_pred, **kwargs)
    return score


mae = partial(scoring, score_func=l1_loss)
mse = partial(scoring, score_func=mse_loss)
r2_score = partial(scoring, score_func=lambda y, y_pred: 1 - mse_loss(y, y_pred)/mse_loss(y, y.mean()))

