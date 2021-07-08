from typing import Dict

import torch as t
import torch.distributions as dist
from nflib.flows import AffineHalfFlow, NormalizingFlowModel
from torch import nn


def make_tensor(val, dtype=t.float32):
    if t.is_tensor(val):
        return val
    else:
        return t.tensor(val, dtype=dtype)


class BaseDist(nn.Module):

    @property
    def dist(self) -> t.distributions.Distribution:
        raise NotImplemented('Please implement distribution access property')

    def get_annotated_params(self) -> Dict[str, float]:
        raise NotImplemented('Please implement annotated params accessor')

    @property
    def name(self) -> str:
        raise NotImplemented('Please implement distribution name accessor')


class DeterministicDist(BaseDist):

    def __init__(self, value):
        super().__init__()
        self._value = nn.Parameter(make_tensor(value), requires_grad=True)

    def __call__(self, batch_shape: t.Size):
        return self._value.repeat(batch_shape)

    @property
    def dist(self) -> t.distributions.Distribution:
        raise ValueError('Distribution is not defined for deterministic value')

    def get_annotated_params(self) -> Dict[str, float]:
        return {
            'mu': self._value.item()
        }

    @property
    def name(self) -> str:
        return 'deterministic'


class NormalDist(BaseDist):

    def __init__(self, mu, sigma):
        super().__init__()
        self._mu = nn.Parameter(make_tensor(mu), requires_grad=True)
        self._sigma = nn.Parameter(make_tensor(sigma), requires_grad=True)
        self._dist = dist.Normal(self._mu, self._sigma)

    def __call__(self, batch_shape: t.Size):
        return self._dist.rsample(batch_shape)

    @property
    def dist(self) -> t.distributions.Distribution:
        return self._dist

    def get_annotated_params(self) -> Dict[str, float]:
        return {
            'mu': self._mu.item(),
            'sigma': self._sigma.item()
        }

    @property
    def name(self) -> str:
        return 'normal'


class GaussianMixtureDist(BaseDist):

    def __init__(self, mu, sigma, mu2, sigma2):
        super().__init__()
        self._mu = nn.Parameter(make_tensor([mu, mu2]), requires_grad=True)
        self._sigma = nn.Parameter(make_tensor([sigma, sigma2]), requires_grad=True)
        mixture = dist.Categorical(t.ones(2, ))
        components = dist.Normal(self._mu, self._sigma)
        self._dist = dist.MixtureSameFamily(mixture, components)

    def __call__(self, batch_shape: t.Size):
        return self._dist.sample(batch_shape)

    @property
    def dist(self) -> t.distributions.Distribution:
        return self._dist

    def get_annotated_params(self) -> Dict[str, float]:
        return {
            'mu': self._mu[0].item(),
            'sigma': self._sigma[0].item(),
            'mu2': self._mu[1].item(),
            'sigma2': self._sigma[1].item()
        }

    @property
    def name(self) -> str:
        return 'gaussian mixture'


# https://github.com/karpathy/pytorch-normalizing-flows
class RealNVP(BaseDist):

    def __init__(
            self,
            dim: int,
            num_transforms: int = 9,
    ):
        super().__init__()
        prior = dist.TransformedDistribution(dist.Uniform(t.zeros(dim), t.ones(dim)), dist.SigmoidTransform().inv)
        flows = [AffineHalfFlow(dim=dim, parity=i % 2) for i in range(num_transforms)]
        self._model = NormalizingFlowModel(prior, flows)

    def __call__(self, batch_shape: t.Size):
        return self._model.sample(batch_shape)

    # @property
    # def dist(self) -> t.distributions.Distribution:
    #     return self._dist

    def get_annotated_params(self) -> Dict[str, float]:
        return {}

    @property
    def name(self) -> str:
        return 'real nvp'
