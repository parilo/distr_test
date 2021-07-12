from typing import Optional, Dict

import torch
import torch.distributions as dist
from torch import nn

from distr_test.distr import BaseDist


class ConditionedAffineHalfFlow(nn.Module):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """
    def __init__(self, parity, dim=None, s_cond=None, t_cond=None):
        super().__init__()
        self.dim = dim
        self.parity = parity
        self.s_cond = s_cond if s_cond is not None else (lambda x: x.new_zeros(x.size(0), self.dim // 2))
        self.t_cond = t_cond if t_cond is not None else (lambda x: x.new_zeros(x.size(0), self.dim // 2))

    def forward(self, x, condition):
        x0, x1 = x[: ,::2], x[: ,1::2]
        if self.parity:
            x0, x1 = x1, x0
        s = self.s_cond(torch.cat([x0, condition], dim=-1))
        t = self.t_cond(torch.cat([x0, condition], dim=-1))
        z0 = x0 # untouched half
        z1 = torch.exp(s) * x1 + t # transform this half as a function of the other
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det

    def backward(self, z, condition):
        z0, z1 = z[: ,::2], z[: ,1::2]
        if self.parity:
            z0, z1 = z1, z0
        s = self.s_cond(torch.cat([z0, condition], dim=-1))
        t = self.t_cond(torch.cat([z0, condition], dim=-1))
        x0 = z0 # this was the same
        x1 = (z1 - t) * torch.exp(-s) # reverse the transform on this half
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det


class ConditionedNormalizingFlow(nn.Module):
    """ A sequence of Normalizing Flows is a Normalizing Flow """

    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x, condition):
        m, _ = x.shape
        log_det = torch.zeros(m)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x, condition)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z, condition):
        m, _ = z.shape
        log_det = torch.zeros(m)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z, condition)
            log_det += ld
            xs.append(z)
        return xs, log_det


class ConditionedNormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """

    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flow = ConditionedNormalizingFlow(flows)

    def forward(self, x, condition):
        zs, log_det = self.flow.forward(x, condition)
        try:
            prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        except ValueError as e:
            print(f'--- value error: {e}')
            print(zs[-1])
            raise e
        return zs, prior_logprob, log_det

    def backward(self, z, condition):
        xs, log_det = self.flow.backward(z, condition)
        return xs, log_det

    def sample(self, num_samples, condition):
        z = self.prior.sample((num_samples,))
        xs, _ = self.flow.backward(z, condition)
        return xs


# https://github.com/karpathy/pytorch-normalizing-flows
class ConditionedRealNVP(BaseDist):

    def __init__(
            self,
            dim: int,
            num_transforms: int = 9,
            s_module: Optional[nn.Module] = None, use factories to have different modules in each flow
            t_module: Optional[nn.Module] = None,
    ):
        super().__init__()
        prior = dist.TransformedDistribution(
            dist.Uniform(
                torch.zeros(dim),
                torch.ones(dim)
            ),
            dist.SigmoidTransform().inv
        )
        # prior = dist.Normal(torch.zeros(dim), torch.ones(dim))
        flows = [
            ConditionedAffineHalfFlow(
                dim=dim,
                parity=i % 2,
                s_cond=s_module,
                t_cond=t_module,
            ) for i in range(num_transforms)
        ]
        self._model = ConditionedNormalizingFlowModel(prior, flows)

    def __call__(self, batch_shape: torch.Size, condition: torch.Tensor):
        assert len(batch_shape) == 1, 'Only shapes with length == 1 supported, shapes like (n,)'
        return self._model.sample(batch_shape[0], condition)[-1]

    def log_prob(self, batch: torch.Tensor, condition: torch.Tensor):
        _, prior_logprob, log_det = self._model.forward(batch, condition)
        return prior_logprob  # + log_det

    def get_annotated_params(self) -> Dict[str, float]:
        return {}

    @property
    def name(self) -> str:
        return 'conditioned real nvp'
