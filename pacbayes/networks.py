import numpy as np
import torch
import torch.nn as nn
import math
from .utils import (
    init_sequential_weights,
    init_layer_weights,
    pad_concat,
    compute_dists,
    compute_dists_2d,
    device,
    Identity
)

def set_grad(feature_map, requires_grad):
    for param in feature_map.parameters():
        param.requires_grad = requires_grad


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, width: int = 50,
                 num_layers: int =2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nonlinearity = torch.relu
        self.layers = self._initialize_layers(num_layers, width)

    def _initialize_layers(self, num_layers: int, width: int) -> nn.ModuleList:
        # num_layers = number of hidden layers
        layers = nn.ModuleList([])
        for i in range(num_layers + 1):
            input_dim = self.input_dim if i == 0 else width
            output_dim = self.output_dim if i == num_layers else width
            layer = nn.Linear(input_dim, output_dim)
            layers.append(layer)
        return layers

    def forward(self, x):
        # inputs are [N, D]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.nonlinearity(x)
        return x


class MLPFeatureMap(MLP):
    def __init__(self, input_dim, feature_dim, num_layers, width):
        super().__init__(input_dim=input_dim,
                         output_dim=feature_dim,
                         width=width,
                         num_layers=num_layers)
        self.feature_dim = feature_dim


class ImageFeaturesMap(nn.Module):
    def __init__(self, feature_dim, init_length_scale=.05,
                 learn_length_scale=True, points_per_axis=28, init_PCA=False):
        """
        Args:
            num_features: int
            init_length_scale: float, initial lengthscale of RBFs
            learn_length_scale: bool, whether to learn the lengthscale
            points_per_axis: int, number of RBFs per dimension of image
        """
        super().__init__()

        self.feature_dim = feature_dim
        sigma = torch.Tensor([math.log(init_length_scale)]).to(device)
        self.sigma = nn.Parameter(sigma)
        self.sigma.requires_grad = learn_length_scale
        self.sigma_fn = torch.exp
        self.points_per_axis = points_per_axis

        # Initialise the image-features
        if init_PCA:  # Use PCA to initialise the features. MNIST only for now.
            assert self.points_per_axis == 28 # This was resolution used for PCA.
            pca_path = 'mnist/data/pca_components.npy'
            pca_components = np.load(pca_path)  # [num_components=512, 28, 28]

            # This flipping and permuting necessary to get the image features
            # in the right orientation.
            pca_components = np.flip(pca_components, axis=1).copy()
            image_features = torch.Tensor(pca_components[:feature_dim, :, :])  # [M, P, P]
            image_features = image_features.permute(0, 2, 1)  # [M, P, P]
        else: # Random init.
            image_features = torch.randn(feature_dim,
                                         points_per_axis,
                                         points_per_axis) / math.sqrt(feature_dim)
        self.image_features = nn.Parameter(image_features.to(device))  # [M, P, P]

        grid = torch.linspace(1e-3, 1. - 1e-3, points_per_axis).to(device)  # [P]
        self.grid = grid[None, :, None]  # [1, P, 1]

    def forward(self, x):
        """
        Args:
            x: [B, N, 2] torch, inputs of observations, assumed 2D
        Returns:
            [B, N, feature_dim] torch, features associated with each input.
        """
        assert x.shape[-1] == 2

        x0, x1 = x[:, :, 0:1], x[:, :, 1:2]  # [B, N, 1]
        dists_0 = compute_dists(self.grid, x0)[:, None, :, :]  # [B, 1, P, N]
        dists_1 = compute_dists(self.grid, x1)[:, None, :, :]  # [B, 1, P, N]
        k0 = self.rbf(dists_0)  # [B, 1, P, N]
        k1 = self.rbf(dists_1)  # [B, 1, P, N]
        image_features = self.image_features[None, :, :, :]  # [1, M, P, P]
        features = torch.sum(k0 * (image_features @ k1), dim=-2)  # [B, M, N] sum over P
        features = features.permute(0, 2, 1)  # [B, N, M]
        return features

    def rbf(self, dists):
        # Compute the RBF kernel, broadcasting appropriately.
        lengthscale = self.sigma_fn(self.sigma)
        return torch.exp(-0.5 * dists / lengthscale ** 2)


class RBFFeatureMap(nn.Module):
    def __init__(self, x_grid, init_length_scale=.05,
                 learn_length_scale=True):
        super().__init__()
        """RBF feature map, spaced on a grid.
        Args:
            x_grid: [num_features, input_dim] torch, locations of RBFs.
            init_length_scale: float, initial length scale of RBFs.
            learn_length_scale: bool, whether to learn length scale.
            input_dim: int, number of input dimensions, 1 or 2.
        """
        sigma = torch.Tensor([math.log(init_length_scale)])
        self.sigma = nn.Parameter(sigma)
        self.sigma.requires_grad = learn_length_scale
        self.sigma_fn = torch.exp
        self.x_grid = x_grid  # [M, input_dim]
        assert len(x_grid.shape) == 2
        self.input_dim = x_grid.shape[-1]
        self.dist_function = compute_dists if self.input_dim == 1 \
            else compute_dists_2d

    def forward(self, x):
        """Forward pass through the layer with evaluations at locations `t`.
        Args:
            x : [B, N, input_dim] torch, inputs of observations
        Returns:
            [B, N, feature_dim] torch, RBF features associated with each input.
        """
        # Compute the pairwise distances.
        x_grid = self.x_grid[None, :, :].repeat(x.shape[0], 1, 1)  # [B, feature_dim, input_dim]
        dists = self.dist_function(x, x_grid)  # [B, N, feature_dim].
        features = self.rbf(dists)
        return features

    def rbf(self, dists):
        """Compute the RBF values for the distances using the correct length
        scales.
        Args:
            dists: [B, N, feature_dim] torch, Pair-wise distances between
                `x` and `x_grid`.
        Returns:
            [B, N, feature_dim] torch, Evaluation of `psi(x, t)` with `psi`
                an RBF kernel.
        """
        # Compute the RBF kernel, broadcasting appropriately.
        lengthscale = self.sigma_fn(self.sigma)
        return torch.exp(-0.5 * dists / lengthscale ** 2)


class SimpleConv(nn.Module):
    """This is a 4-layer convolutional network with fixed stride and channels,
    using ReLU activations.
    Args:
        in_channels: int, Number of channels on the input to the
            network. Defaults to 8.
        out_channels: int, Number of channels on the output by the
            network. Defaults to 8.
    """

    def __init__(self, channels=8):
        super(SimpleConv, self).__init__()
        self.in_channels = channels
        self.out_channels = channels
        self.activation = nn.ReLU()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=channels,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=channels, out_channels=channels,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=channels, out_channels=channels,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=channels, out_channels=channels,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        init_sequential_weights(self.conv_net)
        self.num_halving_layers = 0

    def forward(self, x):
        """Forward pass through the convolutional structure.
        Args:
            x: [B, N, in_channels] torch, Inputs.
        Returns:
            tensor: [B, N_out, out_channels] torch, Outputs.
        """
        return self.conv_net(x)


class UNet(nn.Module):
    """This is a 12-layer residual network with skip connections implemented by
    concatenation.
    Args:
        in_channels: int, Number of channels on the input to network. Defaults
            to 8.
    """

    def __init__(self, channels=8):
        super(UNet, self).__init__()
        self.activation = nn.ReLU()
        self.in_channels = channels
        self.out_channels = 2 * channels  # Because of concatenation.
        self.num_halving_layers = 6

        self.l1 = nn.Conv1d(in_channels=channels,
                            out_channels=channels,
                            kernel_size=5, stride=2, padding=2)
        self.l2 = nn.Conv1d(in_channels=channels,
                            out_channels=2 * channels,
                            kernel_size=5, stride=2, padding=2)
        self.l3 = nn.Conv1d(in_channels=2 * channels,
                            out_channels=2 * channels,
                            kernel_size=5, stride=2, padding=2)
        self.l4 = nn.Conv1d(in_channels=2 * channels,
                            out_channels=4 * channels,
                            kernel_size=5, stride=2, padding=2)
        self.l5 = nn.Conv1d(in_channels=4 * channels,
                            out_channels=4 * channels,
                            kernel_size=5, stride=2, padding=2)
        self.l6 = nn.Conv1d(in_channels=4 * channels,
                            out_channels=8 * channels,
                            kernel_size=5, stride=2, padding=2)

        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6]:
            init_layer_weights(layer)

        self.l7 = nn.ConvTranspose1d(in_channels=8 * channels,
                                     out_channels=4 * channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l8 = nn.ConvTranspose1d(in_channels=8 * channels,
                                     out_channels=4 * channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l9 = nn.ConvTranspose1d(in_channels=8 * channels,
                                     out_channels=2 * channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l10 = nn.ConvTranspose1d(in_channels=4 * channels,
                                      out_channels=2 * channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)
        self.l11 = nn.ConvTranspose1d(in_channels=4 * channels,
                                      out_channels=channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)
        self.l12 = nn.ConvTranspose1d(in_channels=2 * channels,
                                      out_channels=channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)

        for layer in [self.l7, self.l8, self.l9, self.l10, self.l11, self.l12]:
            init_layer_weights(layer)

    def forward(self, x):
        """Forward pass through the convolutional structure.
        Args:
            x: [B, N, in_channels] torch, Inputs.
        Returns:
            tensor: [B, N_out, out_channels] torch, Outputs.
        """
        h1 = self.activation(self.l1(x))
        h2 = self.activation(self.l2(h1))
        h3 = self.activation(self.l3(h2))
        h4 = self.activation(self.l4(h3))
        h5 = self.activation(self.l5(h4))
        h6 = self.activation(self.l6(h5))
        h7 = self.activation(self.l7(h6))

        h7 = pad_concat(h5, h7)
        h8 = self.activation(self.l8(h7))
        h8 = pad_concat(h4, h8)
        h9 = self.activation(self.l9(h8))
        h9 = pad_concat(h3, h9)
        h10 = self.activation(self.l10(h9))
        h10 = pad_concat(h2, h10)
        h11 = self.activation(self.l11(h10))
        h11 = pad_concat(h1, h11)
        h12 = self.activation(self.l12(h11))

        return pad_concat(x, h12)


"""
    Depthwise Separable Convs
"""


class SeparableConv2d(nn.Module):
    """Implementation of a depthwise separable convolution for 1d signals
    Args:
        in_channels (int): number of channels of incoming signal
        out_channels (int): number of channels for outgoing signal
        kernel_size (int): width of convolutional filter
        stride (int): stride of convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(SeparableConv2d, self).__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(in_channels=in_channels,
                                   out_channels=in_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   stride=stride,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   padding=0,
                                   stride=1)

    def forward(self, x):
        """Forward pass through the depthwise separable layer
        Args:
            x (torch.tensor): batch x in_channels x L_in tensor of input
            signals
        Returns:
            (torch.tensor): batch x out_channels x L_out tensor of outputs
            signals
        """
        x = self.depthwise(x)
        return self.pointwise(x)


"""
    2D Architectures
"""


class ConvNet2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        super(ConvNet2d, self).__init__()

        self.input_dimensionality = 2
        self.activation = nn.ReLU()
        self.conv_net = self.construct_architecture()

    def construct_architecture(self):
        return nn.Sequential(
            Identity()
        )

    def forward(self, x):
        return self.conv_net(x)


class NoConv(ConvNet2d):
    def __init__(self, in_channels=4, out_channels=4):
        super(NoConv, self).__init__(in_channels, out_channels)
        self.num_halving_layers = 0


class SimpleConv2d(ConvNet2d):
    def __init__(self, in_channels=8, out_channels=8):
        super(SimpleConv2d, self).__init__(in_channels, out_channels)
        self.num_halving_layers = 0

    def construct_architecture(self):
        model = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=self.out_channels,
                      kernel_size=5, stride=1, padding=2),
        )
        init_sequential_weights(model)
        return model


class SimpleSeparableConv2d(ConvNet2d):
    def __init__(self, in_channels=8, out_channels=8):
        super(SimpleSeparableConv2d, self).__init__(in_channels, out_channels)
        self.num_halving_layers = 0

    def construct_architecture(self):
        model = nn.Sequential(
            SeparableConv2d(in_channels=self.in_channels, out_channels=64,
                            kernel_size=5, stride=1),
            nn.ReLU(),
            SeparableConv2d(in_channels=64, out_channels=64,
                            kernel_size=5, stride=1),
            nn.ReLU(),
            SeparableConv2d(in_channels=64, out_channels=64,
                            kernel_size=5, stride=1),
            nn.ReLU(),
            SeparableConv2d(in_channels=64, out_channels=self.out_channels,
                            kernel_size=5, stride=1),
        )
        init_sequential_weights(model)
        return model


class SimpleConv2dXL(ConvNet2d):
    def __init__(self, in_channels=8, out_channels=8):
        super(SimpleConv2dXL, self).__init__(in_channels, out_channels)
        self.num_halving_layers = 0

    def construct_architecture(self):
        model = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=8,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=8,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=self.out_channels,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        init_sequential_weights(model)
        return model


class HourGlassConv2d(ConvNet2d):
    def __init__(self, in_channels=8, out_channels=8):
        super(HourGlassConv2d, self).__init__(in_channels, out_channels)
        self.num_halving_layers = 2

    def construct_architecture(self):
        model = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=16,
                      kernel_size=5, stride=2, padding=2),
            self.activation,
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=2, padding=2),
            self.activation,
            nn.ConvTranspose1d(in_channels=32, out_channels=16,
                               kernel_size=5, stride=2, padding=2,
                               output_padding=1),
            self.activation,
            nn.ConvTranspose1d(in_channels=16, out_channels=self.out_channels,
                               kernel_size=5, stride=2, padding=2,
                               output_padding=1),
            self.activation
        )
        init_sequential_weights(model)
        return model
