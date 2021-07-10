from typing import Dict, Tuple

import torch as t
import torch.optim as optim
import torch.nn.functional as F

from distr import BaseDist


class DistrOptimizer:

    def __init__(self, distr: BaseDist, lr: float = 1e-3):
        self._distr = distr
        self._opt = optim.Adam(self._distr.parameters(), lr=lr)

    def loss(self, batch: t.Tensor) -> Tuple[t.Tensor, Dict[str, float]]:
        raise NotImplemented('Please implement loss function calculation')

    def optimize(self, loss: t.Tensor):
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()

    def train_step(self, batch: t.Tensor) -> Dict[str, float]:
        loss, log_data = self.loss(batch)
        self.optimize(loss)
        return log_data


class DistMSEOptimizer(DistrOptimizer):

    def loss(self, batch: t.Tensor):
        prediction = self._distr(batch.shape)
        loss = F.mse_loss(prediction, batch)

        log_data = {
            'mse_loss': loss.item(),
        }
        return loss, log_data


class DistLogPOptimizer(DistrOptimizer):

    def loss(self, batch: t.Tensor):
        logp = self._distr.log_prob(batch)
        loss = -logp.mean()

        log_data = {
            'logp_loss': loss.item(),
        }
        return loss, log_data
