import torch


def r2_score(y_true, y_pred):
    with torch.no_grad():
        y_true = y_true.to(y_pred.device)
        rss = torch.nn.functional.mse_loss(y_true, y_pred.view(y_true.shape), reduction='sum').item()
        tss = torch.nn.functional.mse_loss(y_true, y_true.mean(), reduction='sum').item()
    return 1 - rss / tss
