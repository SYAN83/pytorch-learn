import time
from torch.utils.data import TensorDataset, DataLoader
from ..base import BaseEstimator
from ..utils import set_device, preprocess_data


class SGDBaseEstimator(BaseEstimator):

    model = None
    optimizer = None
    loss_fn = None
    losses = None

    def __init__(self, fit_intercept=True, batch_size=-1, shuffle=True, max_iter=1000, device='cuda:0', verbose=True):
        self.fit_intercept = fit_intercept
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_iter = max_iter
        self.verbose = verbose
        self.device = set_device(device, verbose=verbose)

    def _fit(self, X, y):
        X = X.to(device=self.device)
        y = y.to(device=self.device, dtype=X.dtype).view(-1, 1)
        dataset = TensorDataset(X, y)

        batch_size = X.shape[0] if self.batch_size < -1 else self.batch_size
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=self.shuffle)
        self.losses = list()
        start_time = time.time()
        for epoch in range(self.max_iter):
            epoch_loss_tot = 0
            for X_, y_ in dataloader:
                y_pred = self.model(X_)
                loss = self.loss_fn(y_pred, y_)
                epoch_loss_tot += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                self.losses.append(epoch_loss_tot/len(dataset))
                if self.verbose and epoch % (self.max_iter // 20) == 0:
                    print('[Epoch {}/{}]] Loss: {:.2f}'.format(epoch,
                                                               self.max_iter,
                                                               self.losses[-1]))
        if self.verbose:
            print('Total running time: {:.2f}s'.format(time.time() - start_time))

        return self

    def predict(self, X):
        y_pred = self.model.eval()(X.to(device=self.device))
        return y_pred

