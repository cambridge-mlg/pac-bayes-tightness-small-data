import numpy as np
import math
import torch
import torch.nn as nn

from .utils import (
    device,
    init_sequential_weights,
    compute_dists,
    compute_dists_2d
)
from .networks import MLP


class DeepSets(nn.Module):
    def __init__(self,
                 x_dim=1,
                 y_dim=1,
                 output_dim=50,
                 rep_dim=50,
                 enc_layers=1,
                 enc_width=50,
                 dec_layers=1,
                 dec_width=50):
        super().__init__()
        self.output_dim = output_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.encoder = MLP(input_dim=x_dim + y_dim,
                           output_dim=rep_dim,
                           width=enc_width,
                           num_layers=enc_layers)
        self.decoder = MLP(input_dim=rep_dim,
                           output_dim=output_dim,
                           width=dec_width,
                           num_layers=dec_layers)

    def forward(self, x, y):
        """
        Args:
            x: [B, N, x_dim] torch, input datapoints
            y: [B, N, y_dim] torch, output datapoints

        Returns:
            output: [B, output_dim] torch, DeepSets function of the datasets.
        """
        batch_size = x.shape[0]
        N = x.shape[1]
        if N == 0:  # Empty context set.
            rep_dim = self.encoder.output_dim
            reps = torch.ones(batch_size, rep_dim).to(device) / math.sqrt(rep_dim)
            return self.decoder(reps)  # [B, output_dim]

        # Encodergit stat
        xy = torch.cat((x, y), dim=-1)  # [B, N, x_dim + y_dim]
        xy = xy.view(-1, self.x_dim + self.y_dim) # [B * N, x_dim + y_dim]
        reps = self.encoder(xy)  # [B * N, rep_dim]

        # Aggregate representations
        reps = reps.view(batch_size, N, -1)  # [B, N, rep_dim]
        reps = reps.mean(dim=1)  # [B, rep_dim] Average over data points

        # Decoder
        return self.decoder(reps)  # [B, output_dim]


class ConvDeepSet(nn.Module):
    """One-dimensional set convolution layer. Uses an RBF kernel for
    `psi(x, x')`.
    Args:
        in_channels: int, Number of input channels.
        out_channels: int, Number of output channels.
        learn_length_scale: bool, Learn the length scales of the channels.
        init_length_scale: float, Initial value for the length scale.
        use_density: bool, optional, Append density channel to inputs.
            Defaults to `True`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 learn_length_scale,
                 init_length_scale,
                 use_density=True):
        super(ConvDeepSet, self).__init__()
        self.out_channels = out_channels
        self.use_density = use_density
        self.in_channels = in_channels + 1 if self.use_density else in_channels
        self.g = self.build_weight_model()
        self.sigma = nn.Parameter(np.log(init_length_scale) *
                                  torch.ones(self.in_channels),
                                  requires_grad=learn_length_scale)
        self.sigma_fn = torch.exp

    def build_weight_model(self):
        """Returns a function point-wise function that transforms the
        `in_channels + 1`-dimensional representation to dimensionality
        `out_channels`.
        Returns:
            :class:`torch.nn.Module`: Linear layer applied point-wise to
                channels.
        """
        model = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
        )
        init_sequential_weights(model)
        return model

    def forward(self, x, y, t):
        """Forward pass through the layer with evaluations at locations `t`.
        Args:
            x : [B, N, 1] torch, inputs of observations
            y : [B, N, in_channels] torch, Outputs of observations.
            t : [B, N_target, 1] torch, Inputs to evaluate function at.
        Returns:
            [B, N_target, out_channels] torch, Outputs of evaluated function at
                `z`.
        """
        # Ensure that `x`, `y`, and `t` are rank-3 tensors.
        if len(x.shape) == 2:
            x = x.unsqueeze(2)
        if len(y.shape) == 2:
            y = y.unsqueeze(2)
        if len(t.shape) == 2:
            t = t.unsqueeze(2)

        # Compute shapes.
        batch_size = x.shape[0]
        n_in = x.shape[1]
        n_out = t.shape[1]

        # Compute the pairwise distances.
        # Shape: (batch, n_in, n_out).
        dists = compute_dists(x, t)

        # Compute the weights.
        # Shape: (batch, n_in, n_out, in_channels).
        wt = self.rbf(dists)

        if self.use_density:
            # Compute the extra density channel.
            # Shape: (batch, n_in, 1).
            density = torch.ones(batch_size, n_in, 1).to(device)

            # Concatenate the channel.
            # Shape: (batch, n_in, in_channels).
            y_out = torch.cat([density, y], dim=2)
        else:
            y_out = y

        # Perform the weighting.
        # Shape: (batch, n_in, n_out, in_channels).
        y_out = y_out.view(batch_size, n_in, -1, self.in_channels) * wt

        # Sum over the inputs.
        # Shape: (batch, n_out, in_channels).
        y_out = y_out.sum(1)

        if self.use_density:
            # Use density channel to normalize convolution
            density, conv = y_out[..., :1], y_out[..., 1:]
            normalized_conv = conv / (density + 1e-8)
            y_out = torch.cat((density, normalized_conv), dim=-1)

        # Apply the point-wise function.
        # Shape: (batch, n_out, out_channels).
        y_out = y_out.view(batch_size * n_out, self.in_channels)
        y_out = self.g(y_out)
        y_out = y_out.view(batch_size, n_out, self.out_channels)

        return y_out

    def rbf(self, dists):
        """Compute the RBF values for the distances using the correct length
        scales.
        Args:
            dists: torch, Pair-wise distances between `x` and `t`.
        Returns:
            torch, Evaluation of `psi(x, t)` with `psi` an RBF kernel.
        """
        # Compute the RBF kernel, broadcasting appropriately.
        scales = self.sigma_fn(self.sigma)[None, None, None, :]
        a, b, c = dists.shape
        return torch.exp(-0.5 * dists.view(a, b, c, -1) / scales ** 2)


class SetConv2D(nn.Module):
    """2-dimensional Set convolution layer. Uses an RBF kernel for psi(x, x').
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        learn_length_scale (bool): Learn the length scales of the channels.
        init_length_scale (float): Initial value for the length scale.
        use_density (bool, optional): Append density channel to inputs.
        (Default: True)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 learn_length_scale,
                 init_length_scale,
                 use_density=True):
        super(SetConv2D, self).__init__()
        self.out_channels = out_channels
        self.use_density = use_density
        self.in_channels = in_channels + 1 if self.use_density else in_channels
        self.g = self.build_weight_model()
        self.sigma = nn.Parameter(np.log(init_length_scale) *
                                  torch.ones(self.in_channels),
                                  requires_grad=learn_length_scale)
        self.sigma_fn = torch.exp

    def build_weight_model(self):
        """Returns a function point-wise function that transforms the
            (in_channels + 1)-dimensional representation to dimensionality
            out_channels.
        Returns:
            torch.nn.Module: Linear layer applied point-wise to channels.
        """
        model = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
        )
        init_sequential_weights(model)
        return model

    def forward(self, x, y, t):
        """Forward pass through the layer with evaluations at locations t
        Args:
            x (torch.tensor): n x 2 tensor of input observation locations
            y (torch.tensor): n x in_channels tensor of input observation
            values
            t (torch.tensor): m x 2 input locations to evaluate computed
            function
        Returns:
            torch.tensor: m x out_channels tensor of output function values
        """
        # Ensure that `x`, `y`, and `t` are rank-3 tensors.
        assert len(x.shape) == 3 and \
               len(y.shape) == 3 and \
               len(t.shape) == 3, 'x, y, and t must be 3-tensors'

        # Compute shapes.
        batch_size = x.shape[0]
        n_in = x.shape[1]
        n_out = t.shape[1]

        # Compute the pairwise distances.
        # Shape: batch x n_in x n_out.
        dists = compute_dists_2d(x, t)

        # Compute the weights.
        # Shape: batch x n_in x n_out x channel_in.
        wt = self.rbf(dists)

        if self.use_density:
            # Compute the extra density channel.
            # Shape: batch x n_in x 1.
            density = torch.ones(batch_size, n_in, 1).to(device)

            # Concatenate the channel.
            # Shape: batch x n_in x (channel_in).
            y_out = torch.cat([density, y], dim=2)
        else:
            y_out = y

        # Perform the weighting.
        # Shape: batch x n_in x n_out x (channel_in).
        y_out = y_out.view(batch_size, n_in, -1, self.in_channels) * wt

        # Sum over the inputs.
        # Shape: batch x n_out x (channel_in).
        y_out = y_out.sum(1)

        if self.use_density:
            # Use density channel to normalize convolution
            density, conv = y_out[..., :1], y_out[..., 1:]
            normalized_conv = conv / (density + 1e-8)
            y_out = torch.cat((density, normalized_conv), dim=-1)

        # Apply the point-wise function.
        # Shape: batch x n_out x channel_out.
        y_out = y_out.view(batch_size * n_out, self.in_channels)
        y_out = self.g(y_out)
        y_out = y_out.view(batch_size, n_out, self.out_channels)

        return y_out

    def rbf(self, dists):
        """Compute the RBF values for the distances using correct
        length-scales.
        Args:
            dists (torch.tensor): n x m tensor of pair-wise distances between
            x and t.
        Returns:
            (torch.tensor): n x m tensor of psi(x, t) with RBF function
        """
        # Compute the RBF kernel, broadcasting appropriately.
        scales = self.sigma_fn(self.sigma)[None, None, None, :]
        a, b, c = dists.shape
        return torch.exp(-0.5 * dists.view(a, b, c, -1) / scales ** 2)


class ConvCNP(nn.Module):
    """One-dimensional ConvCNP model.
    Args:
        learn_length_scale: bool, Learn the length scale.
        points_per_unit: int, Number of points per unit interval on input.
            Used to discretize function.
        architecture: class:`nn.Module`: Convolutional architecture to place
            on functional representation (rho).
        x_grid: [N_grid] torch, locations of RBF features.
    """

    def __init__(self, learn_length_scale, points_per_unit, cnn, x_grid):
        super(ConvCNP, self).__init__()
        self.activation = nn.Sigmoid()
        self.conv_net = cnn

        # Compute initialisation.
        self.points_per_unit = points_per_unit
        init_length_scale = 2.0 / self.points_per_unit

        self.l0 = ConvDeepSet(
            in_channels=1,
            out_channels=self.conv_net.in_channels,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=True
        )
        self.mean_layer = ConvDeepSet(
            in_channels=self.conv_net.out_channels,
            out_channels=1,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=False
        )
        self.sigma_layer = ConvDeepSet(
            in_channels=self.conv_net.out_channels,
            out_channels=1,
            learn_length_scale=learn_length_scale,
            init_length_scale=init_length_scale,
            use_density=False
        )

        # [num_points, input_dim]
        self.x_grid = x_grid
        self.num_points = x_grid.shape[0]

    def forward(self, x, y):
        """Run the model forward.
        Args:
            x : [B, N, input_dim] torch, Observation locations.
            y : [B, N, input_dim] torch, Observation values.
        Returns:
            dict[tensor]: Means and variances of shape
                [B, N_grid].
        """
        # Ensure that `x`, `y`, and `t` are rank-3 tensors.
        assert len(x.shape) == 3
        assert len(y.shape) == 3

        # [B, N, input_dim]
        x_grid = self.x_grid[None, :, :].repeat(x.shape[0], 1, 1)

        # Apply first layer and conv net. Take care to put the axis ranging
        # over the data last.
        h = self.activation(self.l0(x, y, x_grid))  # [B, N_grid, cnn_in_channels]
        h = h.permute(0, 2, 1)  # [B, cnn_in_channels, N_grid]
        h = h.reshape(h.shape[0], h.shape[1], self.num_points)
        h = self.conv_net(h)  # [B, cnn_out_channels, N_grid]
        h = h.reshape(h.shape[0], h.shape[1], -1).permute(0, 2, 1)

        # Check that shape is still fine!
        if h.shape[1] != x_grid.shape[1]:
            raise RuntimeError('Shape changed.')

        # Produce means and standard deviations. Always evaluate the output on
        # the same grid.
        means = self.mean_layer(x_grid, h, x_grid)  # [B, N_grid, 1]
        var_params = self.sigma_layer(x_grid, h, x_grid)  # [B, N_grid, 1]

        means = means[..., 0]   # [B, N_grid]
        var_params = var_params[..., 0]  # [B, N_grid]
        return torch.cat([means, var_params], dim=1)  # [B, 2 * N_grid]

    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])


class ConvCNP2D(nn.Module):
    """2-dimensional convolutional inference network.
    Args:
        latent_dim (int): number of channels in latent function
        learn_length_scale (bool): length-scale parameter for
            set-conv layers is learnable.
        points_per_unit (int): density with which to discretize
            latent function
        architecture (nn.Sequential): convolutional architecture
        latent_locations (torch.tensor): locations in latent space
            for latent function discretization
    """

    def __init__(self,
                 learn_length_scale,
                 points_per_unit,
                 cnn,
                 x_grid):
        super().__init__()

        self.in_channels = 1
        self.out_channels = 1
        self.activation = nn.Sigmoid()
        self.conv_net = cnn
        self.x_grid = x_grid
        self.multiplier = 2 ** self.conv_net.num_halving_layers
        self.input_dimensionality = self.conv_net.input_dimensionality

        # Compute initialisation.
        # self.dim_density = points_per_unit ** (1.0 / self.input_dimensionality)
        self.init_length_scale = 1.0 / points_per_unit
        # Initialize input output layers
        self.input_layer = SetConv2D(in_channels=self.in_channels,
                                     out_channels=self.conv_net.in_channels,
                                     learn_length_scale=learn_length_scale,
                                     init_length_scale=self.init_length_scale,
                                     use_density=True)

        self.mean_layer = SetConv2D(in_channels=self.conv_net.out_channels,
                                    out_channels=self.out_channels,
                                    learn_length_scale=learn_length_scale,
                                    init_length_scale=self.init_length_scale,
                                    use_density=False)

        self.sigma_layer = SetConv2D(in_channels=self.conv_net.out_channels,
                                     out_channels=self.out_channels,
                                     learn_length_scale=learn_length_scale,
                                     init_length_scale=self.init_length_scale,
                                     use_density=False)


    def forward(self, x, y):
        """Run inference for a given dataset.
        Args:
            x (tensor): Observation locations of shape
            `batch x data x features`.
            y (tensor): Observation values of shape
            `batch x data x outputs`.
        Returns:
            tuple[tensor]: Means and standard deviations of shape
                `batch_out x channels_out`.
        """
        batch_size, num_context = x.shape[0], x.shape[1]
        x_grid = self.x_grid[None, :, :].repeat(batch_size, 1, 1)  # [B, P^2, 2]
        assert len(x_grid.shape) == 3

        num_per_dim = int(math.sqrt(x_grid.shape[1]))
        assert num_per_dim ** 2 == x_grid.shape[1]
        num_per_dim = [num_per_dim, num_per_dim]  # Square grid.

        # Apply set-conv and permute for convolutional architecture
        # dim(h) = batch_size x num_grid_locations x n_channels
        h = self.activation(self.input_layer(x, y, x_grid))

        # Reshape inputs into image shape, and transpose dimensions
        # dim(h) = batch_size x n_grid1 x n_grid0 x n_channels
        h = h.view(batch_size, *num_per_dim, self.input_layer.out_channels)

        # Channels first: dim(h) = batch_size x n_channels x grid1 x grid0
        h = h.permute(0, 3, 1, 2)

        # image-shaped: dim(h) = batch_size x n_channels x grid0 x grid1
        h = h.transpose(2, 3)

        # Pass through convnet
        # dim(h) = batch_size x out_channels x grid0 x grid1
        h = self.conv_net(h)

        # Go back to set shaped
        h = h.transpose(2, 3)
        h = h.permute(0, 2, 3, 1)
        h = h.reshape(batch_size, -1, self.conv_net.out_channels)

        # Check that shape is still fine!
        if h.shape[1] != x_grid.shape[1]:
            raise RuntimeError('Shape changed.')

        # Produce means and standard deviations. Always evaluate the output on
        # the same grid.
        means = self.mean_layer(x_grid, h, x_grid)  # [B, N_grid=P^2, 1]
        var_params = self.sigma_layer(x_grid, h, x_grid)  # [B, N_grid=P^2, 1]

        means = means[..., 0]   # [B, N_grid]
        var_params = var_params[..., 0]  # [B, N_grid]
        return torch.cat([means, var_params], dim=1)  # [B, 2 * N_grid]
