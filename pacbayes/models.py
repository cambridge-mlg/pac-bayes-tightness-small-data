import math

import lab.torch as B
import torch
import torch.nn as nn
from scipy.special import loggamma

from .classifiers import ValClassifier, PACBayesClassifier, PACBayesDDPClassifier
from .convex import Convex, BernoulliKL
from .utils import device, log_1_minus_exp_minus


class HoeffValClassifier(ValClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def risk_bounds(self, val_risk):
        """Compute one-sided Hoeffding risk bounds based on validation risks"""
        gap = torch.log(1 / self.delta) / (2 * self.N_val)
        gap = torch.sqrt(gap)
        return val_risk + gap  # [B]


class KLValClassifier(ValClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kl_delta = BernoulliKL()

    def risk_bounds(self, val_risk):
        """Compute the average one-sided KL-Chernoff bound based on validation
        risks"""
        one_sided_c = torch.log(1 / self.delta) / self.N_val
        return self.kl_delta.biggest_inverse(val_risk, one_sided_c)


class CatoniMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.freeze_beta = False

    def set_beta(self, freeze_beta, freeze_beta_val=None):
        if freeze_beta:
            assert freeze_beta_val is not None
            self.freeze_beta = True
            self.frozen_beta = torch.Tensor([freeze_beta_val]).to(device)
        else:
            self.freeze_beta = False

    @property
    def beta(self):
        if self.freeze_beta:
            return self.frozen_beta
        else:
            return torch.exp(self.beta_param)

    def risk_bounds(self, emp_risk, KL, N):
        """Compute the Catoni risk bound."""
        log_bounds = self.batch_losses(emp_risk, KL, N)  # [B]
        return torch.exp(log_bounds)

    def batch_losses(self, emp_risk, KL, N):
        """Compute the log of Catoni risk bound, in a numerically stable way."""
        B = self.beta * emp_risk + (KL + torch.log(1 / self.delta)) / N  # [B]
        if self.loss_fn == 'nll' or self.freeze_beta:
            return B  # If beta is frozen, can just minimise linear bound.
        log_bounds = log_1_minus_exp_minus(B) - log_1_minus_exp_minus(self.beta)
        return log_bounds


class CatoniAmortisedBetaMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def beta(self):
        return self.architecture.beta

    def risk_bounds(self, emp_risk, KL, N):
        """Compute the Catoni risk bound."""
        log_bounds = self.batch_losses(emp_risk, KL, N)  # [B]
        return torch.exp(log_bounds)

    def batch_losses(self, emp_risk, KL, N):
        """Compute the log of Catoni risk bound, in a numerically stable way."""
        B = self.beta * emp_risk + (KL + torch.log(1 / self.delta)) / N  # [B]
        if self.loss_fn == 'nll':
            return B
        log_bounds = log_1_minus_exp_minus(B) - log_1_minus_exp_minus(self.beta)
        return log_bounds


class MaurerMixin:
    def __init__(self, optimistic=False, **kwargs):
        super().__init__(**kwargs)
        self.kl_delta = BernoulliKL()
        self.optimistic = optimistic
        self.log_I_term = None
        self.log_I_N = None

    def risk_bounds(self, emp_risk, KL, N):
        """Compute the Maurer risk bound."""
        if self.optimistic:
            rhs = (KL + torch.log(1 / self.delta)) / N
        else:
            rhs = (KL + torch.log(1. / self.delta) + self.compute_log_I_term(N)) / N
        return self.kl_delta.biggest_inverse(emp_risk, rhs)  # [B]

    def batch_losses(self, emp_risk, KL, N):
        """Compute the log of piecewise bound described in Dziugaite et al 2020
        Theorem D.2"""
        if self.optimistic:
            B = (KL + torch.log(1 / self.delta)) / N  # [B]
        else:
            B = (KL + torch.log(1. / self.delta) + self.compute_log_I_term(N)) / N  # [B]
        pinsker_gap = torch.sqrt(B / 2.0)  # [B]
        if self.loss_fn == 'nll':
            return emp_risk + pinsker_gap  # Regression risk may be negative.
        quad_gap = B + torch.sqrt(B * (B + 2.0 * emp_risk))  # [B]
        gap = torch.minimum(quad_gap, pinsker_gap)
        return torch.log(emp_risk + gap)  # [B]. Take logs. Later averaged via logsumexp.

    def compute_log_I_term(self, N):
        """Compute the log I_Delta term for Bernoulli KL Delta."""
        if self.log_I_term is None:  # Compute and store.
            self.log_I_N = N
            log_terms = []
            for k in range(0, N + 1):
                log_term = loggamma(N + 1) - loggamma(N - k + 1) - loggamma(k + 1)
                r = k / N
                if r > 0:
                    log_term += k * B.log(r)
                if r < 1:
                    log_term += (N - k) * B.log(1 - r)
                log_terms.append(log_term)
            self.log_I_term =  B.logsumexp(B.stack(*log_terms))
        else:
            assert self.log_I_N == N  # Make sure N hasn't changed so log_I is still valid.

        return self.log_I_term


class MaurerInvMixin:
    def __init__(self, optimistic=False, **kwargs):
        super().__init__(**kwargs)
        self.optimistic = optimistic
        self.kl_delta = BernoulliKL()
        self.log_I_term = None
        self.log_I_N = None

    def risk_bounds(self, emp_risk, KL, N):
        """Compute the Maurer risk bound."""
        if self.optimistic:  # Illegal Maurer bound without I_Delta term.
            rhs = (KL + torch.log(1. / self.delta)) / N
        else:  # Begin bound with I_Delta term.
            rhs = (KL + torch.log(1. / self.delta) + self.compute_log_I_term(N)) / N
        return self.kl_delta.biggest_inverse(emp_risk, rhs)  # [B]

    def batch_losses(self, emp_risk, KL, N):
        """Directly backprop through the inverse KL."""
        if self.loss_fn == 'nll':  # NLL loss init.
            B = (KL + torch.log(2.0 * math.sqrt(N) / self.delta)) / N  # [B]
            pinsker_gap = torch.sqrt(B / 2.0)  # [B]
            return emp_risk + pinsker_gap  # Regression risk may be negative.
        return torch.log(self.risk_bounds(emp_risk, KL, N))  # Take logs. Later averaged via logsumexp.

    def compute_log_I_term(self, N):
        """Compute the log I_Delta term for Bernoulli KL Delta."""
        if self.log_I_term is None:  # Compute and store.
            self.log_I_N = N
            log_terms = []
            for k in range(0, N + 1):
                log_term = loggamma(N + 1) - loggamma(N - k + 1) - loggamma(k + 1)
                r = k / N
                if r > 0:
                    log_term += k * B.log(r)
                if r < 1:
                    log_term += (N - k) * B.log(1 - r)
                log_terms.append(log_term)
            self.log_I_term =  B.logsumexp(B.stack(*log_terms))
        else:
            assert self.log_I_N == N  # Make sure N hasn't changed so log_I is still valid.

        return self.log_I_term


class ConvexMixin:
    """A mix-in for delta parametrised by a sum of a convex function and a linear
    function.

    Args:
        n_supremum (int, optional): Number of values to use when estimating the
            supremum. Defaults to `10000`.
        epsilon_supremum (scalar, optional): Distance from zero and one to start and
            end the supremum at. Defaults to `1e-6`.
    """

    def __init__(self, n_supremum=10000, epsilon_supremum=1e-6, separable=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_supremum = n_supremum
        self.epsilon_supremum = epsilon_supremum
        self.separable = separable

        if separable:
            self.convex_pop_risk = Convex(n_input=1).to(device)
            self.convex_emp_risk = Convex(n_input=1,
                                          initialise_linear=True).to(device)
        else:
            self.convex_delta = Convex(n_input=2).to(device)

    def compute_supremum(self, n):
        # Estimate the supremum by taking the maximum over a dense `linspace`.
        r = torch.linspace(
            self.epsilon_supremum, 1 - self.epsilon_supremum, self.n_supremum
        ).to(device)
        k_over_ms = torch.linspace(0, 1, n + 1).to(device)  # [n + 1]

        # Precompute the values for delta in the terms of the sum. We want to keep
        # the computation graph lean.
        if self.separable:
            convex_emp_risks = self.convex_emp_risk(k_over_ms)  # [n + 1]
            convex_pop_risks = self.convex_pop_risk(r)  # [R]
            # [n + 1, R]
            convex_risks = convex_emp_risks[:, None] + convex_pop_risks[None, :]
        else:
            convex_risks = B.reshape(self.convex_delta(B.stack(
                B.reshape(B.tile(k_over_ms[:, None], 1, self.n_supremum), -1),
                B.reshape(B.tile(r[None, :], n + 1, 1), -1),
                axis=1
            )), n + 1, self.n_supremum)  # [n + 1, R]

        log_terms = []
        for k in range(0, n + 1):
            logcomb = loggamma(n + 1) - loggamma(n - k + 1) - loggamma(k + 1)
            log_pmf = logcomb + k * B.log(r) + (n - k) * B.log(1 - r)
            delta = convex_risks[k, :]
            log_terms.append(log_pmf + n * delta)
        log_supremum_r = torch.logsumexp(B.stack(*log_terms, axis=1), dim=1)

        return log_supremum_r

    def risk_bounds(self, emp_risk, kl, n):
        log_supremum_r = self.compute_supremum(n)
        bound_on_delta = (kl + torch.max(log_supremum_r) - B.log(self.delta)) / n
        if self.separable:
            bound_on_convex_pop_risk = bound_on_delta - self.convex_emp_risk(emp_risk)
            return self.convex_pop_risk.biggest_inverse(bound_on_convex_pop_risk)
        else:
            return self.convex_delta.biggest_inverse(emp_risk, bound_on_delta)

    def batch_losses(self, emp_risk, kl, n):
        if self.loss_fn == 'nll':
            raise NotImplementedError
        return torch.log(self.risk_bounds(emp_risk, kl, n))  # Take logs. Later averaged via logsumexp.

"""Catoni"""

class CatoniClassifier(CatoniMixin, PACBayesClassifier):
    def __init__(self, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        beta_param = torch.Tensor([math.log(beta)]).to(device)
        self.beta_param = nn.Parameter(beta_param)


class CatoniDDPClassifier(CatoniMixin, PACBayesDDPClassifier):
    def __init__(self, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        beta_param = torch.Tensor([math.log(beta)]).to(device)
        self.beta_param = nn.Parameter(beta_param)


class CatoniAmortisedBetaClassifier(CatoniAmortisedBetaMixin, PACBayesDDPClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


"""Maurer"""

class MaurerClassifier(MaurerMixin, PACBayesClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MaurerDDPClassifier(MaurerMixin, PACBayesDDPClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MaurerInvDDPClassifier(MaurerInvMixin, PACBayesDDPClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MaurerInvClassifier(MaurerInvMixin, PACBayesClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

"""Learned convex"""

class ConvexClassifier(ConvexMixin, PACBayesClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ConvexDDPClassifier(ConvexMixin, PACBayesDDPClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

