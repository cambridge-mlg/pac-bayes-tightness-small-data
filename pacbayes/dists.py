import math

import lab.torch as B
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence

from .utils import device


class Dist(nn.Module):
    def __init__(self, is_network_parameter=False, **params):
        """
        Args:
            network_parameter: bool, whether to treat the means and variances
                as torch Parameters. If True, these will be optimised along
                with the rest of the model.
            parameters: [B, ...] torch tensors, batched parameters of the
                distribution.
        """
        super().__init__()
        # Check the batch shapes are consistent.
        self.batch_size = next(iter(params.values())).shape[0]
        for param in params.values():
            assert param.shape[0] == self.batch_size
        self.is_network_parameter = is_network_parameter
        self.learnable = False

        # Store the parameters.
        self.params = params
        if is_network_parameter:
            # Learn these parameters as part of the network at meta-train time.
            # Need to use ParameterDict so these parameters are registered.
            registered_params = nn.ParameterDict()
            for key, param in self.params.items():
                registered_params[key] = nn.Parameter(param)
            self.params = registered_params

    def set_learnable(self):
        """Only for post optimisation."""
        assert not self.is_network_parameter, (
            "Network parameters are already " "learnable."
        )

        for key, param in self.params.items():
            self.params[key] = Variable(param.detach(), requires_grad=True)

        self.learnable = True

    def learnable_param_list(self):
        """Only for post optimisation."""
        assert not self.is_network_parameter
        assert self.learnable == True
        return [param for param in self.params.values()]

    def fix_learned_params(self):
        """Only for post optimisation."""
        assert not self.is_network_parameter
        assert self.learnable == True

        for key, param in self.params.items():
            self.params[key] = param.detach()

        self.learnable = False


class FullCovGaussianFamily(nn.Module):
    def __init__(self):
        super().__init__()

    def instantiate_dist(self, feature_dim, batch_size=1, is_network_parameter=False):
        mean_init = torch.zeros([batch_size, feature_dim]).to(device)
        num_chol_params = int(feature_dim * (feature_dim + 1) / 2)
        # chol_params can take any sign.
        chol_params_init = 1e-2 * torch.randn([batch_size, num_chol_params]).to(device)
        return Dist(
            means=mean_init,
            chol_params=chol_params_init,
            is_network_parameter=is_network_parameter,
        )

    def num_param_dims(self, feature_dim):
        """Dimension of parameters of distribution in the mean-field Gaussian
        family. Means and cholesky params."""
        return int(feature_dim + feature_dim * (feature_dim + 1) / 2)

    def predictive(self, post, feature_map, x):
        """Compute predictive distribution of function.
        Args:
            post: dist object, batch of B
            x: [B, N, input_dim] torch, input datapoints

        Returns:
            pred_means: [B, N] torch, mean of predictive function at x
            pred_vars: [B, N] torch, var of predictive function at x
        """
        # Get posterior and features
        features = feature_map(x)  # [B, N, D]
        D = features.shape[-1]

        # Compute predictive mean and variance
        post_means = post.params["means"]  # [B, D]
        post_chols = self.param_vector_to_chol(post.params["chol_params"])  # [B, D, D]

        post_means = post_means[..., None]  # [B, D, 1]
        pred_means = torch.matmul(features, post_means)  # [B, N, 1]
        pred_means = pred_means[..., 0]  # [B, N]

        # Sigma = LL^T, Var = (phi^T L)(L^T phi) = || phi^T L ||^2
        phi_T_chols = torch.matmul(features, post_chols)  # [B, N, D]
        pred_vars = torch.sum(phi_T_chols**2, dim=2)  # [B, N]

        pred_vars = torch.clamp(pred_vars, min=1e-6)
        return pred_means, pred_vars

    def kl(self, q, p):
        """
        Args:
            q: dist object, batch B
            p: dist object, batch B

        Returns:
            KLs: [B] torch, KL(Q || P).
        """
        p_means = p.params["means"]  # [B, D]
        D = p_means.shape[1]
        p_chols = self.param_vector_to_chol(p.params["chol_params"])  # [B, D, D]
        normal_p = MultivariateNormal(loc=p_means, scale_tril=p_chols)

        q_means = q.params["means"]  # [B, D]
        assert D == q_means.shape[1], "Need matching feature dims"
        q_chols = self.param_vector_to_chol(q.params["chol_params"])  # [B, D, D]
        normal_q = MultivariateNormal(loc=q_means, scale_tril=q_chols)

        return kl_divergence(normal_q, normal_p)  # [B]

    def param_vector_to_dist(self, param_vector, feature_dim):
        """
        Args:
            param_vector: [B, D + D * (D + 1) / 2] torch, where D = feature_dim.
                Elements could take any sign.
        Returns:
            Dist object.
        """
        means = param_vector[:, :feature_dim]  # [B, D]
        chol_params = param_vector[:, feature_dim:]  # [B, D * (D + 1) / 2]
        assert chol_params.shape[1] == int(feature_dim * (feature_dim + 1) / 2)

        return Dist(means=means, chol_params=chol_params)

    def mean_cov_to_dist(self, means, covs):
        """
        Args:
            means: [B, D] torch, mean vectors
            chols: [B, D, D] torch, covariance matrices.

        Returns:
            Dist object.
        """
        chols = torch.cholesky(covs)
        chol_params = self.chol_to_param_vector(chols)

        return Dist(means=means, chol_params=chol_params)

    def param_vector_to_cov(self, param_vectors):
        """
        Args:
            param_vectors: [B, D * (D + 1) / 2] torch, Cholesky parameter
                vectors. Values can take any sign.

        Returns:
            covs: [B, D, D] torch, covariance matrices.
        """
        chols = self.param_vector_to_chol(param_vectors)
        covs = chols @ chols.transpose(1, 2)
        return covs

    def param_vector_to_chol(self, param_vectors):
        """
        Args:
            param_vectors: [B, D * (D + 1) / 2] torch, Cholesky parameter
                vectors. Values can take any sign.

        Returns:
            chols: [B, D, D] torch, Cholesky factors
        """
        assert len(param_vectors.shape) == 2, "Check for batch dimension"

        n_params = param_vectors.shape[-1]
        D = round((-1 + math.sqrt(1 + 8 * n_params)) / 2)

        diags = param_vectors[:, :D]  # [B, D]
        off_diags = param_vectors[:, D:]  # [B, D * (D - 1) / 2]

        with B.on_device(str(device)):
            chols = B.vec_to_tril(off_diags, offset=-1)  # [B, D, D]

        # Exponentiate diagonals
        chols.diagonal(dim1=-2, dim2=-1)[:] += torch.exp(diags)

        return chols

    def chol_to_param_vector(self, chols):
        """
        Args:
            chols: [B, D, D] torch, Cholesky factors.

        Returns:
            param_vectors: [B, D * (D + 1) / 2] torch, Cholesky parameter
                vectors. Values can take any sign.
        """
        diags = torch.log(chols.diagonal(dim1=-2, dim2=-1)[:])  # [B, D]
        with B.on_device(str(device)):
            off_diags = B.tril_to_vec(chols, offset=-1)  # [B, D * (D - 1) / 2]

        return torch.cat([diags, off_diags], dim=-1)


class MeanFieldGaussianFamily(nn.Module):
    def __init__(self):
        super().__init__()

    def instantiate_dist(self, feature_dim, batch_size=1, is_network_parameter=False):
        mean_init = torch.zeros([batch_size, feature_dim]).to(device)
        var_init = torch.ones_like(mean_init)
        return Dist(
            means=mean_init, vars=var_init, is_network_parameter=is_network_parameter
        )

    def num_param_dims(self, feature_dim):
        """Dimension of parameters of distribution in the mean-field Gaussian
        family. Factor of 2 since there are means and variances."""
        return 2 * feature_dim

    def predictive(self, post, feature_map, x):
        """Compute predictive distribution of function.
        Args:
            post: dist object, batch of B
            x: [B, N, input_dim] torch, input datapoints

        Returns:
            pred_means: [B, N] torch, mean of predictive function at x
            pred_vars: [B, N] torch, var of predictive function at x
        """
        # Get posterior and features
        features = feature_map(x)  # [B, N, feature_dim]

        # Compute predictive mean and variance
        post_means = post.params["means"]
        post_vars = post.params["vars"]

        post_means = post_means[..., None]  # [B, feature_dim, 1]
        pred_means = torch.matmul(features, post_means)  # [B, N, 1]
        pred_means = pred_means[..., 0]  # [B, N]

        post_vars = post_vars[..., None]  # [B, feature_dim, 1]
        pred_vars = torch.matmul(features**2, post_vars)  # [B, N, 1]
        pred_vars = pred_vars[..., 0]  # [B, N]

        pred_vars = torch.clamp(pred_vars, min=1e-6)
        return pred_means, pred_vars

    def kl(self, q, p):
        """
        Args:
            q: dist object, batch B
            p: dist object, batch B

        Returns:
            KLs: [B] torch, KL(Q || P).
        """
        p_means = p.params["means"]
        p_vars = p.params["vars"]
        q_means = q.params["means"]
        q_vars = q.params["vars"]

        log_term = torch.log(p_vars) - torch.log(q_vars)
        quad_term = (q_vars + (q_means - p_means) ** 2) / p_vars
        return 0.5 * torch.sum(log_term + quad_term - 1.0, dim=-1)

    def param_vector_to_dist(self, param_vector, feature_dim):
        """
        Args:
            param_vector: [B, 2 * feature_dim] torch

        Returns:
            MeanFieldGaussian object
        """
        means = param_vector[:, :feature_dim]  # [B, feature_dim]
        vars = param_vector[:, feature_dim:]  # [B, feature_dim]
        vars = torch.exp(vars)

        assert means.shape[-1] == feature_dim
        assert vars.shape[-1] == feature_dim

        return Dist(means=means, vars=vars)
