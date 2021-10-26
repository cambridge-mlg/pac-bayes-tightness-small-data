import logging

import lab.torch as B
import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from plum import Dispatcher
from torch.optim import Adam

from .utils import device, bern_kl

__all__ = ["BernoulliKL", "CatoniMixture", "Convex", "differentiable_biggest_inverse"]


dispatch = Dispatcher()

log = logging.getLogger(__name__)


def _ensure_torch_float(x, at_least_vector=False):
    if isinstance(x, int):
        x *= 1.0
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if B.isscalar(x) and at_least_vector:
        x = x[None]
    return x


class Delta(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def derivative(self, x):
        """Derivative of the convex function.

        Args:
            x (vector): Inputs to compute derivative at.

        Returns:
            vector: Derivatives.
        """
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            return torch.autograd.grad(B.sum(self(x)), x)[0]

    @dispatch
    def biggest_inverse(self, x1: B.TorchNumeric, c: B.TorchNumeric, **kw_args):
        """Biggest inverse.

        Further takes in keyword arguments for :func:`.differentiable_biggest_inverse`.

        Args:
            x1 (vector): Values for first input.
            c (vector): Outputs to compute biggest inverse for.

        Returns:
            vector: Biggest inverses.
        """
        def f(theta_, x_):
            return self(B.stack(theta_, x_, axis=1))

        def f_detached(theta_, x_):
            # `x_` will be a scalar when the upper bound is dynamically expanded. If
            # `x_` is a scalar, `B.stack` will fail, so convert it to a vector in
            # that case.
            if B.isscalar(x_):
                x_ = B.ones(theta_).to(device) * x_
            x_ = B.stack(theta_, x_, axis=1)
            return self(x_).detach()

        def df_detached(theta_, x_):
            x_ = B.stack(theta_, x_, axis=1)
            return self.derivative(x_)[:, 1].detach()

        return differentiable_biggest_inverse(
            f, f_detached, df_detached, c, x1, **kw_args
        )

    @dispatch
    def biggest_inverse(self, c: B.TorchNumeric, **kw_args):
        """Biggest inverse.

        Further takes in keyword arguments for :func:`.differentiable_biggest_inverse`.

        Args:
            c (vector): Outputs to compute biggest inverse for.

        Returns:
            vector: Biggest inverses.
        """
        def f_detached(x):
            return self(x).detach()

        def df_detached(x):
            return self.derivative(x).detach()

        return differentiable_biggest_inverse(
            self, f_detached, df_detached, c, **kw_args
        )


class PositiveLinear(nn.Module):
    """A linear layer with positive weights.

    Args:
        in_features (int): Input dimensionality.
        out_features (int): Output dimensionality.
        bias (bool, optional): Bias. Must be `False`. Defaults to `False`.
    """

    def __init__(self, in_features, out_features, bias=False):
        if bias:
            raise NotImplementedError("Bias not supported.")

        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        """Randomly initialise the parameters of this layer."""
        nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        return F.linear(x, torch.exp(self.weight) / self.in_features)


class BernoulliKL(Delta):
    """Bernoulli KL delta."""

    def forward(self, x):
        q, p = x[:, 0], x[:, 1]
        return bern_kl(q, p)


class CatoniMixture(Delta):
    """Bernoulli KL delta.

    Args:
        cs (vector): Catoni parameters.
        ws (vector, optional): Positive weights. Defaults to the uniform distribution
            over `cs`.
        learn (bool, optional): Learn the parameters. Defaults to `True`.

    """

    def __init__(self, cs, ws=None, learn=True):
        super(CatoniMixture, self).__init__()
        cs = _ensure_torch_float(cs, at_least_vector=True)
        if ws is None:
            ws = B.ones(cs) / B.length(cs)
        if learn:
            self.cs = nn.Parameter(cs, requires_grad=learn)
            self.log_ws = nn.Parameter(B.log(ws), requires_grad=learn)
        else:
            self.cs = cs
            self.log_ws = B.log(ws)

    def forward(self, x):
        q, p = x[:, 0:1], x[:, 1:2]
        cs = self.cs[None, :]
        ws = B.exp(self.log_ws[None, :])
        return B.mean(ws * (-q * cs - B.log(p * (B.exp(-cs) - 1) + 1)), axis=1)


class Convex(Delta):
    """A convex function built from a sum of an affine function and a two-linear-layer
    neural network where the activation function is convex and the weights of the
    second linear layer are constrained to be positive. Upon initialisation,
    an optimisation is run to fit the convex function to the Catoni function.

    Args:
        n_hidden (int, optional): Dimensionality of the hidden layer. Defaults to `128`.
        n_input (int, optional): Dimensionality of the inputs. Defaults to `2`.
        initialise_linear (bool, optional): Initialise to the linear Catoni function.
            Defaults to `False`.
        bias (bool, optional): Whether to learn a bias term. Defaults to `False`.
        init_iters (int, optional): Number of ADAM iterations to perform when
            initialising the random convex function to be close to Catoni.
    """

    def __init__(
        self,
        n_hidden=128,
        n_input=2,
        initialise_linear=False,
        bias=False,
        init_iters=5,
    ):
        super(Convex, self).__init__()
        self.n_input = n_input
        self.affine = nn.Linear(n_input, 1, bias=bias)
        self.convex = nn.ModuleList(
            [
                nn.Linear(n_input, n_hidden, bias=True),
                nn.Softplus(),
                PositiveLinear(n_hidden, 1, bias=False),
            ]
        )

        self.init_iters = init_iters

        if n_input == 1:
            if initialise_linear:
                self.fit_catoni_linear()
            else:
                self.fit_catoni()
        if n_input == 2:
            self.fit_catoni_2d()

    def forward(self, x):
        # This function includes some magic constants that make the initialisation more
        # suitable for our purposes.

        # Ensure that `x` is a rank-2 tensor and help initialisation.
        x = 10 * B.uprank(x) - 5

        # Apply layers of convex function.
        res = x
        for layer in self.convex:
            res = layer(res)

        # Add affine function to the convex function.
        res = res + self.affine(x) / 10

        # Squeeze the extra dimension before returning.
        return B.squeeze(res)

    def fit_catoni_2d(self):
        """Fit the 2D, non-separable convex function to the Catoni function."""
        q = torch.rand((1000,))
        p = torch.rand((1000,))
        beta = 2
        catoni = -B.log(1 - p * (1 - B.exp(-beta))) - beta * q
        opt = Adam(self.parameters(), lr=5e-2)
        for i in range(self.init_iters):
            error = catoni - self(B.stack(q, p, axis=1))
            loss = B.mean((error - B.mean(error)) ** 2)
            loss.backward()
            opt.step()
            opt.zero_grad()

    def fit_catoni(self):
        """Fit the convex function to the Catoni function."""
        x = torch.linspace(0, 1, 100)
        catoni = -B.log(1 - x * (1 - B.exp(-2)))
        opt = Adam(self.parameters(), lr=5e-2)
        for i in range(self.init_iters):
            error = catoni - self(x)
            loss = B.mean((error - B.mean(error)) ** 2)
            loss.backward()
            opt.step()
            opt.zero_grad()

    def fit_catoni_linear(self):
        """Fit the convex function to the linear Catoni function."""
        x = torch.linspace(0, 1, 100)
        catoni = -2 * x
        opt = Adam(self.parameters(), lr=5e-2)
        for i in range(self.init_iters):
            error = catoni - self(x)
            loss = B.mean((error - B.mean(error)) ** 2)
            loss.backward()
            opt.step()
            opt.zero_grad()


def _insert_unused_first_argument(f):
    def new_f(_, x):
        return f(x)

    return new_f


def differentiable_biggest_inverse(
    f,
    f_detached,
    df_detached,
    c,
    theta=None,
    lower=0.0,
    upper=1.0,
    upper_max=100.0,
    atol=1e-4,
    n_grid_max=200_000,
):
    """Compute the biggest inverse whilst enabling gradients with PyTorch.

    Args:
        f (function): Function to invert. If `theta` is not given, `f` takes in one
            argument. If `theta is given, then `f` takes in two arguments: the first
            argument is `theta`, a parameter, and the second argument is `x`,
            the input. This function should support batched evaluation and *should*
            carry gradients.
        f_detached (function): Function to invert. This function should support
            batched evaluation but *should not* carry gradients.
        df_detached (function): Derivative of the function to invert. This function
            should support batched evaluation but *should not* carry gradients.
        c (tensor): Outputs to invert.
        theta (tensor, optional): Values of parameters, one for every element in `c`.
        lower (float): Lower bound on the possible inputs that the inverse can yield.
            This bound should always hold. Defaults to `0`.
        upper (float): Upper bound on the possible inputs that the inverse can yield.
            This bound will be dynamically increased if it appears to not hold.
            Defaults to `1.0`.
        upper_max (float): Maximum value for `upper` when `upper` is dynamically
            increased. Defaults to `100.0`.
        atol (scalar): Absolute tolerance. Defaults to `1e-4`. Will not be respected
            if the number of grid points exceeds `n_grid_max`.
        n_grid_max (int): Maximum length of grid. Defaults to `200_000`.

    Returns:
        vector: Biggest inverse of `f` evaluated at all of `c`.
    """
    # If `theta` is not given, reduce to the trivial case where `theta` is always
    # `None`.
    if theta is None:
        f = _insert_unused_first_argument(f)
        f_detached = _insert_unused_first_argument(f_detached)
        df_detached = _insert_unused_first_argument(df_detached)

    dtype = B.dtype(c)
    c_detached = c.detach()
    if theta is None:
        theta_detached = None
        n_theta = None
    else:
        # Ensure that `theta` is a vector.
        if B.isscalar(theta):
            theta = B.expand_dims(theta, axis=0)
        theta_detached = theta.detach()
        n_theta = B.shape(theta)[0]

    # Expand upper bound to exceed all values of `c`.
    c_max = B.max(c_detached)
    while (
        B.min(f_detached(theta_detached, torch.tensor(upper, dtype=dtype).to(device)))
        < c_max
    ):
        upper = upper * 2
        # Don't let this loop go on indefinitely.
        if upper >= upper_max:
            upper = upper_max
            break

    # Generate an appropriate grid.
    n_grid = min(int(np.ceil((upper - lower) / atol)) + 1, n_grid_max)
    grid = torch.linspace(lower, upper, n_grid, dtype=dtype, device=device)

    # Find an upcrossing. There should be at most one upcrossing. If there is no
    # upcrossing, return the last value and kill the gradient.
    if theta is None:
        fs = f_detached(None, grid)[None, :]
    else:
        fs = B.reshape(
            f_detached(
                B.reshape(B.tile(theta_detached[:, None], 1, n_grid), -1),
                B.reshape(B.tile(grid[None, :], n_theta, 1), -1),
            ),
            n_theta,
            n_grid,
        )
    upcrossing = (fs[:, :-1] <= c_detached[:, None]) & (fs[:, 1:] > c_detached[:, None])
    ok = B.any(upcrossing, axis=1)
    inds = torch.argmax(B.cast(torch.int, upcrossing), dim=1)
    inds = ok * inds + ~ok * (n_grid - 1)
    xs = grid[inds]

    # Give a warning if it computation failed for some of the points.
    if B.any(~ok):
        log.warning("Biggest inverse failed for some points.")

    # Trick RMAD into computing the right derivative, but only at the values which
    # are okay. Also, don't just divide by `df_detached`, because that can give a
    # potential division by zero!
    ift = 1 / (1e-20 + df_detached(theta_detached, xs))
    return xs + ok * ift * (c - f(theta, xs))
