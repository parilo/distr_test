from typing import Dict, Tuple, Optional

import torch as t
import torch.optim as optim
import torch.nn.functional as F

from distr_test.distr import BaseDist


class DistrOptimizer:

    def __init__(self, distr: BaseDist, lr: float = 1e-3):
        self._distr = distr
        self._opt = optim.Adam(self._distr.parameters(), lr=lr)

    def loss(self, batch: t.Tensor, condition: Optional[t.Tensor] = None) -> Tuple[t.Tensor, Dict[str, float]]:
        raise NotImplemented('Please implement loss function calculation')

    def optimize(self, loss: t.Tensor):
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()

    def train_step(self, batch: t.Tensor, condition: Optional[t.Tensor] = None) -> Dict[str, float]:
        loss, log_data = self.loss(batch, condition)
        if loss is not None:
            self.optimize(loss)
        return log_data


class DistMSEOptimizer(DistrOptimizer):

    def loss(self, batch: t.Tensor, condition: Optional[t.Tensor] = None):
        if condition is not None:
            prediction = self._distr(batch.shape, condition)
        else:
            prediction = self._distr(batch.shape)
        loss = F.mse_loss(prediction, batch)

        log_data = {
            'mse_loss': loss.item(),
        }
        return loss, log_data


class DistLogPOptimizer(DistrOptimizer):

    def loss(self, batch: t.Tensor, condition: Optional[t.Tensor] = None):
        # try:
        if condition is not None:
            logp = self._distr.log_prob(batch, condition)
        else:
            logp = self._distr.log_prob(batch)
        # except ValueError as e:
        #     print(f'--- value error: {e}')
        #     return None, {'logp_loss': -1}
        loss = -logp.mean()

        log_data = {
            'logp_loss': loss.item(),
        }
        return loss, log_data
