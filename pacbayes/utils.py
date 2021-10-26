import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lab as B
from numbers import Number
from scipy.special import roots_hermitenorm as hermite_nodes
from scipy.special import loggamma


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""Device perform computations on."""


def _e_bar(delta, val_risks, n):
    delta = B.to_numpy(delta)
    n = int(B.to_numpy(n))
    all_k_obs = np.round(B.to_numpy(val_risks) * n).astype(int)
    bounds = []
    for k_obs in all_k_obs:
        p = B.linspace(1e-6, 1 - 1e-6, 10_000)
        log_terms = []
        for k in range(0, k_obs + 1):
            log_comb = loggamma(n + 1) - loggamma(n - k + 1) - loggamma(k + 1)
            log_pmf = log_comb + k * B.log(p) + (n - k) * B.log(1 - p)
            log_terms.append(log_pmf)
        valid = B.logsumexp(B.stack(*log_terms, axis=1), axis=1) <= B.log(delta)
        if not any(valid):
            bounds.append(1)
        else:
            bounds.append(p[np.argmax(valid)])
    return torch.tensor(bounds, device=device)


def set_all_seeds(seed):
    # Set all random seeds.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _to_numpy(x):
    """Convert a PyTorch tensor to NumPy."""
    return x.detach().cpu().numpy()


def _list_to_numpy_squeeze(xs):
    """Convert a list of PyTorch tensors to NumPy, and squeeze"""
    return [ _to_numpy(x).squeeze() for x in xs]


def log_1_minus_exp_minus(x):
    """Compute log(1 - exp(-x)) for x > 0 in a numerically stable way.
    Following implementation from https://github.com/wouterkool/estimating-
    gradients-without-replacement/blob/9d8bf8b/bernoulli/gumbel.py#L7-L11
    which is also mentioned in https://github.com/pytorch/pytorch/issues/39242.
    See also https://cran.r-project.org/web/packages/Rmpfr/vignettes/
    log1mexp-note.pdf
    """
    assert torch.min(x) >= 0.
    x = -x.abs()
    out = torch.where(x > -0.693,
                      torch.log(-torch.expm1(x)),
                      torch.log1p(-torch.exp(x)))
    return out


def bern_kl(q, p):
    """
    Compute KL divergence KL(Ber(q) || Ber(p)), taking care with cases of
    0 * log(0).
    Args:
        q: [B] torch, q values
        p: [B] torch, p values
    Returns:
        kls: [B], torch, KL divergences.
    """
    tol = 1e-6  # tolerance used to define 0 * log(0) cases
    q = torch.clamp(q, min=tol, max=1. - tol)
    p = torch.clamp(p, min=tol, max=1. - tol)

    term1 = q * (torch.log(q) - torch.log(p))
    term2 = (1. - q) * (torch.log(1. - q) - torch.log(1. - p))

    kl = term1 + term2
    kl = torch.clamp(kl, 0., 1e6)

    assert torch.all(kl >= 0.)
    assert torch.all(torch.isfinite(kl))

    return kl


def bern_kl_inv(q, c, iters=15):
    """
    Use Newton's method to estimate
        sup {p in [0,1] : kl(q||p) leq c}
        following the description in Dziugaite and Roy 2017, Appendix A.
        N.B. Care should be taken if q is very close to 1, since then the
        Bernoulli kl is quite ill conditioned as a function of p.
    Args:
        q: [B] torch, q values
        c: [1] torch, c value - N.B. cannot broadcast over c
        iters: int, number of Newton's method iterations to use
    Returns:
        p: [B] torch, estimates of suprema. N.B. returned as a double
    """
    if len(q.shape) == 0:
        q = q[None]  # For broadcasting
    if len(c.shape) == 0:
        c = c[None]
    q = q.type(torch.DoubleTensor)
    c = c.type(torch.DoubleTensor)

    # Initialise Newton with Pinsker upper bound.
    estimate = q + torch.sqrt(c / 2.)
    for _ in range(iters):
        # Stay within [0,1]
        if torch.max(estimate) >= 1.:
            estimate[torch.where(estimate >= 1.)] = 1. - 1e-10
        # Newton iteration
        estimate = estimate - h(q, c, estimate) / h_prime(q, estimate)

    # Check that numerical issues haven't brought it out of [0,1] too far.
    assert torch.max(estimate) <= 1. + 1e-2
    estimate = torch.minimum(estimate, torch.Tensor([1.]))  # sup can be at most 1.
    return estimate


def h(q, c, p):
    """Compute kl(q||p) - c. See Dziugaite and Roy 2017 Appendix A."""
    return bern_kl(q, p) - c


def h_prime(q, p):
    """Compute derivative of h with respect to p. See Dziugaite and Roy 2017
    Appendix A.
    """
    return (1 - q) / (1 - p) - (q / p)


def to_multiple(x, multiple):
    """Convert `x` to the nearest above multiple.
    Args:
        x: number, Number to round up.
        multiple: int, Multiple to round up to.

    Returns:
        number: `x` rounded to the nearest above multiple of `multiple`.
    """
    if x % multiple == 0:
        return x
    else:
        return x + multiple - x % multiple


def init_layer_weights(layer):
    """Initialize the weights of a :class:`nn.Layer` using Glorot
    initialization.
    Args:
        layer: :class:`nn.Module`, Single dense or convolutional layer from
            :mod:`torch.nn`.

    Returns:
        :class:`nn.Module`: Single dense or convolutional layer with
            initialized weights.
    """
    torch.nn.init.xavier_normal_(layer.weight, gain=1)
    torch.nn.init.constant_(layer.bias, 1e-3)


def init_sequential_weights(model, bias=0.0):
    """Initialize the weights of a nn.Sequential model with Glorot
    initialization.
    Args:
        model: :class:`nn.Sequential`, Container for model.
        bias: float, optional, Value for initializing bias terms. Defaults
            to `0.0`.

    Returns:
        (nn.Sequential): model with initialized weights
    """
    for layer in model:
        if hasattr(layer, 'weight'):
            torch.nn.init.xavier_normal_(layer.weight, gain=1)
        if hasattr(layer, 'bias'):
            torch.nn.init.constant_(layer.bias, bias)
    return model


def compute_dists(x, y):
    """Fast computation of pair-wise squared distances for the 1d case.
    Args:
        x: [B, N, 1] torch, Inputs
        y: [B, M, 1] torch, Other inputs

    Returns:
        Pair-wise distances: [B, N, M], torch
    """
    assert x.shape[2] == 1 and y.shape[2] == 1, \
        'The inputs x and y must be 1-dimensional observations.'
    return (x - y.permute(0, 2, 1)) ** 2


def compute_dists_2d(x, t):
    # Compute pairwise squared distances in a batched way.
    # Use the matrix multiplication trick.
    norms2_x = torch.sum(x ** 2, dim=2)[:, :, None]
    norms2_t = torch.sum(t ** 2, dim=2)[:, None, :]
    dists2 = norms2_x + \
             norms2_t - \
             2 * torch.matmul(x, t.permute(0, 2, 1))
    return dists2


def linspace_nd(ranges, nums):
    """Generalisation of `np.linspace` to multiple dimension.
    Args:
        ranges (tuple[tuple[float]]): Range per dimension.
        num (tuple[int]): Number of points per dimension.
    Returns:
        matrix: Points ranging the specified ranges where rows correspond to
            points and columns to dimensions.
    """
    # Handle one-dimensional specification.
    if isinstance(ranges[0], Number):
        ranges = (ranges,)
    if isinstance(nums, Number):
        nums = (nums,)

    # Check that specification is consistent.
    if len(ranges) != len(nums):
        raise ValueError('Specified {} ranges but {} number(s) of points.'
                         ''.format(len(ranges), len(nums)))

    # Create points linearly spaced.
    x_dims = [np.linspace(r[0], r[1], n) for r, n in zip(ranges, nums)]
    return np.dstack(np.meshgrid(*x_dims)).reshape(-1, len(ranges))


def pad_concat(t1, t2):
    """Concat the activations of two layers channel-wise by padding the layer
    with fewer points with zeros.
    Args:
        t1: [B, N_1, C_1] torch, Activations from first layer.
        t2: [B, N_2, C_2] torch, Activations from second layer.
    Returns:
        [B, max(N_1, N_2), C_1 + C_2] torch, Concatenated activations of both
            layers.
    """
    if t1.shape[2] > t2.shape[2]:
        padding = t1.shape[2] - t2.shape[2]
        if padding % 2 == 0:  # Even difference
            t2 = F.pad(t2, (int(padding / 2), int(padding / 2)), 'reflect')
        else:  # Odd difference
            t2 = F.pad(t2, (int((padding - 1) / 2), int((padding + 1) / 2)),
                       'reflect')
    elif t2.shape[2] > t1.shape[2]:
        padding = t2.shape[2] - t1.shape[2]
        if padding % 2 == 0:  # Even difference
            t1 = F.pad(t1, (int(padding / 2), int(padding / 2)), 'reflect')
        else:  # Odd difference
            t1 = F.pad(t1, (int((padding - 1) / 2), int((padding + 1) / 2)),
                       'reflect')

    return torch.cat([t1, t2], dim=1)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class GaussianIntegrator:

    def __init__(self, num_nodes=64):
        self.num_nodes = num_nodes
        nodes, weights = hermite_nodes(num_nodes)
        self.nodes = torch.Tensor(nodes).to(device)
        self.weights = torch.Tensor(weights).to(device)

    def integrate(self, f, mean, std):
        scaled_nodes = self.nodes * std[..., None] + mean[..., None]
        f_nodes = f(scaled_nodes)
        integral = (self.weights * f_nodes).sum(-1)
        # Scipy integrates against exp(-x**2/2), need normalization constant
        factor = math.sqrt(2 * math.pi)
        return integral / factor
