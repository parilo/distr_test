from typing import Dict, List

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

    def __init__(self, mu: List[float], sigma: List[float]):
        super().__init__()
        self.register_parameter('_mu', nn.Parameter(make_tensor(mu), requires_grad=True))
        self.register_parameter('_sigma', nn.Parameter(make_tensor(sigma), requires_grad=True))
        self.register_parameter('_cat_prob', nn.Parameter(t.ones(len(mu), )))
        mixture = dist.Categorical(self._cat_prob)
        components = dist.Normal(self._mu, self._sigma)
        self._dist = dist.MixtureSameFamily(mixture, components)

    def __call__(self, batch_shape: t.Size):
        return self._dist.sample(batch_shape)

    def log_prob(self, batch: t.Tensor):
        return self._dist.log_prob(batch)

    def get_annotated_params(self) -> Dict[str, float]:
        ret = {}
        ret.update({
            f'mu_{ind}': self._mu[ind].item() for ind in range(len(self._mu))
        })
        ret.update({
            f'sigma_{ind}': self._sigma[ind].item() for ind in range(len(self._sigma))
        })
        return ret

    @property
    def name(self) -> str:
        return 'gaussian mixture'


# https://github.com/karpathy/pytorch-normalizing-flows
class RealNVP(BaseDist):

    def __init__(
            self,
            dim: int,
            num_transforms: int = 9,
            hidden_layer_size: int = 24,
    ):
        super().__init__()
        prior = dist.TransformedDistribution(dist.Uniform(t.zeros(dim), t.ones(dim)), dist.SigmoidTransform().inv)
        flows = [AffineHalfFlow(dim=dim, parity=i % 2, nh=hidden_layer_size) for i in range(num_transforms)]
        self._model = NormalizingFlowModel(prior, flows)

    def __call__(self, batch_shape: t.Size):
        assert len(batch_shape) == 1, 'Only shapes with length == 1 supported, shapes like (n,)'
        return self._model.sample(batch_shape[0])[-1]

    def log_prob(self, batch: t.Tensor):
        _, prior_logprob, log_det = self._model(batch)
        return prior_logprob + log_det

    def get_annotated_params(self) -> Dict[str, float]:
        return {}

    @property
    def name(self) -> str:
        return 'real nvp'
