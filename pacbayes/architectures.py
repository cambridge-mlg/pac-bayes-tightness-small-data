import torch
import torch.nn as nn
from .utils import device, to_multiple, linspace_nd
from .set_conv import DeepSets, ConvCNP, ConvCNP2D
from .networks import RBFFeatureMap, SimpleConv
from .dists import MeanFieldGaussianFamily, FullCovGaussianFamily
from .gnp.gnp import GNP
from .gnp.discretisation import Discretisation1d


class GNPArchitecture(nn.Module):
    def __init__(self, x_lims=(-2, 2), points_per_unit=32,
                 internal_multiplier=2):
        """
        Args:
            x_lims:
            points_per_unit: int, number of points per unit for the feature
                map. This is not the same as the *internal* number of points
                per unit, which is this number times a factor.
        """
        super().__init__()
        # Define dist family
        self.dist_family = FullCovGaussianFamily()
        self.predictive = self.dist_family.predictive
        self.kl = self.dist_family.kl
        self.points_per_unit = points_per_unit
        self.internal_points_per_unit = internal_multiplier * points_per_unit

        # Determine the grid on which to evaluate functional representation /
        # place RBF features.
        self.input_dims = 1
        discretiser = Discretisation1d(points_per_unit=points_per_unit,
                                       multiple=1,
                                       margin=0.1)

        # Grid for features, [num_features, input_dim = 1]
        self.x_grid = torch.Tensor(discretiser(*x_lims)[:, None]).to(device)

        self.feature_dim = self.x_grid.shape[0]
        init_lengthscale = 2.0 / self.feature_dim

        self.feature_map = RBFFeatureMap(self.x_grid,
                                         init_length_scale=init_lengthscale)

        # Grid for GNP must be [1, num_features, 1]
        self.inference_network = GNP(x_target=self.x_grid[None, :],
                                  points_per_unit=self.internal_points_per_unit)

    def infer(self, x, y):
        """Wrapper around the inference network. Return mean and covariance
        matrix."""
        means, covs = self.inference_network(x, y)  # [B, D], [B, D, D]
        return self.dist_family.mean_cov_to_dist(means, covs)


class GNPPriorArchitecture(GNPArchitecture):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Grid for GNP must be [1, num_features, 1]
        self.prior_network = GNP(x_target=self.x_grid[None, :],
                                 points_per_unit=self.internal_points_per_unit)

    def prior(self, x, y):
        """Wrapper around the prior network.  Return mean and covariance
        matrix."""
        means, covs = self.prior_network(x, y)  # [B, D], [B, D, D]
        return self.dist_family.mean_cov_to_dist(means, covs)


class ConvDeepSetsArchitecture(nn.Module):
    def __init__(self, input_dims=1, x_lims=(-2.1, 2.1), points_per_unit=32,
                 cnn=SimpleConv(), dist_family=MeanFieldGaussianFamily()):
        super().__init__()
        # Define ConvCNP
        self.Convcnp = ConvCNP if input_dims == 1 else ConvCNP2D

        # Define functions dependent on distribution family
        self.dist_family = dist_family
        self.predictive = dist_family.predictive
        self.kl = dist_family.kl

        # Determine the grid on which to evaluate functional representation /
        # place RBF features.
        self.input_dims = input_dims
        x_min = x_lims[0]
        x_max = x_lims[1]
        multiplier = 2 ** cnn.num_halving_layers
        points_per_dim = int(to_multiple(points_per_unit * (x_max - x_min),
                                              multiplier))
        if input_dims == 1:
            x_grid = torch.linspace(x_min, x_max, points_per_dim).to(device)
            self.x_grid = x_grid[..., None]  # [num_features, input_dim = 1]
            self.feature_dim = points_per_dim
            init_lengthscale = 2.0 / self.feature_dim
        elif input_dims == 2:
            x_ranges = [(x_min, x_max), (x_min, x_max)]
            num_per_dim = [points_per_dim, points_per_dim]
            self.x_grid = torch.Tensor(linspace_nd(x_ranges, num_per_dim)).to(device)

            self.feature_dim = points_per_dim ** 2
            init_lengthscale = 1.0 / points_per_dim
        else:
            raise NotImplementedError

        self.feature_map = RBFFeatureMap(self.x_grid,
                                         init_length_scale=init_lengthscale)

        self.inference_network = self.Convcnp(x_grid=self.x_grid,
                                              cnn=cnn,
                                              learn_length_scale=True,
                                              points_per_unit=points_per_unit)

    def infer(self, x, y):
        """Wrapper around the inference network. Return means and variances."""
        param_vector = self.inference_network(x, y)
        return self.dist_family.param_vector_to_dist(param_vector,
                                                     self.feature_dim)


class ConvDeepSetsPriorArchitecture(ConvDeepSetsArchitecture):
    def __init__(self, input_dims=1, x_lims=(-2.1, 2.1), points_per_unit=32,
                 cnn=SimpleConv(), dist_family=MeanFieldGaussianFamily()):
        super().__init__(input_dims=input_dims, x_lims=x_lims,
                         points_per_unit=points_per_unit,
                         cnn=cnn, dist_family=dist_family)
        self.prior_network = self.Convcnp(x_grid=self.x_grid,
                                          cnn=cnn,
                                          learn_length_scale=True,
                                          points_per_unit=points_per_unit)

    def prior(self, x, y):
        """Wrapper around the prior network. Return means and variances."""
        param_vector = self.prior_network(x, y)
        return self.dist_family.param_vector_to_dist(param_vector,
                                                     self.feature_dim)


class DeepSetsArchitecture(nn.Module):
    def __init__(self,
                 input_dim=1,
                 output_dim=1,
                 feature_map=None,
                 rep_dim=50,
                 enc_layers=1,
                 enc_width=50,
                 dec_layers=1,
                 dec_width=50,
                 dist_family=MeanFieldGaussianFamily()):
        super().__init__()
        # Define functions dependent on distribution family
        self.dist_family = dist_family
        self.predictive = dist_family.predictive
        self.kl = dist_family.kl

        self.feature_dim = feature_map.feature_dim
        self.feature_map = feature_map
        inf_net_out_dim = self._inf_net_output_dim(self.feature_dim)
        self.inference_network = DeepSets(x_dim=input_dim,
                                          y_dim=output_dim,
                                          output_dim=inf_net_out_dim,
                                          rep_dim=rep_dim,
                                          enc_layers=enc_layers,
                                          enc_width=enc_width,
                                          dec_layers=dec_layers,
                                          dec_width=dec_width)

    def _inf_net_output_dim(self, feature_dim):
        """Return the dimensionality of the inference network's output."""
        return self.dist_family.num_param_dims(feature_dim)

    def infer(self, x, y):
        """Wrapper around the inference network. Return dist object."""
        param_vector = self.inference_network(x, y)  # [B, output_dim]
        return self.dist_family.param_vector_to_dist(param_vector,
                                                     self.feature_dim)


class DeepSetsPriorArchitecture(DeepSetsArchitecture):
    def __init__(self,
                 input_dim=1,
                 output_dim=1,
                 feature_map=None,
                 rep_dim=50,
                 enc_layers=1,
                 enc_width=50,
                 dec_layers=1,
                 dec_width=50,
                 dist_family=MeanFieldGaussianFamily()):
        super().__init__(input_dim=input_dim,
                         output_dim=output_dim,
                         feature_map=feature_map,
                         rep_dim=rep_dim,
                         enc_layers=enc_layers,
                         enc_width=enc_width,
                         dec_layers=dec_layers,
                         dec_width=dec_width,
                         dist_family=dist_family)
        prior_net_out_dim = self._prior_net_output_dim(feature_map.feature_dim)
        self.prior_network = DeepSets(x_dim=input_dim,
                                      y_dim=output_dim,
                                      output_dim=prior_net_out_dim,
                                      rep_dim=rep_dim,
                                      enc_layers=enc_layers,
                                      enc_width=enc_width,
                                      dec_layers=dec_layers,
                                      dec_width=dec_width)

    def _prior_net_output_dim(self, feature_dim):
        """Return the dimensionality of the prior network's output. Here
        we assume the prior network only outputs the mean and variance over
        the feature weights. This method is overridden in classifiers where
        this is not the case."""
        return self.dist_family.num_param_dims(feature_dim)

    def prior(self, x, y):
        """Wrapper around the prior network. Returns dist object."""
        param_vector = self.prior_network(x, y)  # [B, output_dim]
        return self.dist_family.param_vector_to_dist(param_vector,
                                                     self.feature_dim)


class DeepSetsPriorAmortisedBeta(DeepSetsPriorArchitecture):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prior_net_output_dim(self, feature_dim):
        """Return the dimensionality of the prior network's output. Here
        we add an extra dimension for the amortised beta."""
        return self.dist_family.num_param_dims(feature_dim) + 1

    def prior(self, x, y):
        """Wrapper around the prior network. Returns dist object, and
        also sets the current values of beta"""
        outputs = self.prior_network(x, y)  # [B, output_dim]

        betas = outputs[:, -1]  # [B]
        self.beta = torch.exp(betas)  # set value of beta to most recent prior

        param_vector = outputs[:, :-1]  # [B, output_dim - 1]
        return self.dist_family.param_vector_to_dist(param_vector,
                                                     self.feature_dim)
