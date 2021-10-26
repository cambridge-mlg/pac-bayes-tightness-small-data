import abc
import os
import pickle
from pathlib import Path
from random import shuffle

import numpy as np
import stheno
from scipy.stats import bernoulli

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .utils import device

# ORO_MEAN, ORO_STD = 2266.437744140625, 3559.019287109375
# T2M_MEAN, T2M_STD = 284.766998291015, 6.137677192687988
# # N.B. TP statistics computed only on a subset since there is masking.
# TP_MEAN, TP_STD = 7.268712943186983 * 1e-05, 0.00026009161956608295

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _rand(val_range, *shape):
    lower, upper = val_range
    return lower + np.random.rand(*shape) * (upper - lower)


def _uprank(a):
    if len(a.shape) == 1:
        return a[:, None, None]
    elif len(a.shape) == 2:
        return a[:, :, None]
    elif len(a.shape) == 3:
        return a
    else:
        return ValueError(f"Incorrect rank {len(a.shape)}.")


def get_coords(input, H_pixels, W_pixels):
    """Convert x, y in real plane to pixel locations"""
    # Scale up to [0, H] x [0, W], shape [B, N, 2]
    xs, ys = input[:, :, 0], input[:, :, 1]
    xs = torch.floor(xs * W_pixels)  # [0, W-1], shape [B, N]
    ys = torch.floor((1. - ys) * H_pixels)  # [0, H-1], shape [B, N]
    coords = torch.stack([xs, ys], dim=2)  # [B, N, 2]
    coords = coords.long()
    return coords

def index_into_image(images, coords, channel):
    """
    Args:
        images: [B, C, H, W] torch, images
        coords: [B, N, 2] torch, input indices
        channel: int, channel to index into
    Returns:
        image_vals: [B, N] torch, indexed values
    """
    batch_size = coords.shape[0]
    num_datapoints = coords.shape[1]
    target_images = images[:, channel, :, :]

    image_vals = torch.zeros(batch_size, num_datapoints)  # [B, N]
    for i in range(batch_size):
        # y coordinate gets indexed first since images are H x W
        image_vals[i, :] = target_images[i, coords[i, :, 1], coords[i, :, 0]]  # [N]
    return image_vals  # [B, N]


def sample_from_image(images, batch_size, num_datapoints, dist='uniform',
                      gaussian_std=.2, continuous_output=False,
                      target_channel=0, input_channels=None,
                      forced_input_locs=None):
    """
    Converts a batch of images on the grid, to a set of continuously sampled
    inputs and the corresponding set of outputs.

    Args:
        images: [B, C, H, W] torch. NB: EXPECT VALUES TO BE IN [0, 1].
        batch_size: int, number of images in a batch
        num_datapoints: int, number of datapoints sampled from each image.
        dist: str, whether to use uniform or gaussian input sampling.
        gaussian_std: float, std deviation of gaussian input distribution.
        continuous_output: bool, whether to provide a regression target instead
            of a classification one.
        target_channel: int, the index of the channel whose value we are
            targeting. For MNIST, this is just 0.
        input_channels: list of ints, indeces of the channels whose values
            are appended to the input locations x, y.
        forced_input_locs: [B, N, 2] torch, if set, the image must be sampled
            from this position.

    Returns:
        inputs: [B, N, 2] torch if the images have a single channel. Else
            additional input channels are appended to the dimension, to get
            [B, N, 2 + num_additional_inputs] shape.
        outputs: [B, N, 1] torch. Assume that we only ever predict a single
            scalar output.
    """
    input_batch = batch_size

    # Sample the inputs, [B, N, 2]
    if forced_input_locs is not None:
        inputs = forced_input_locs.cpu()  # Data here hasn't been moved to CUDA yet
    elif dist == 'uniform':
        inputs = torch.rand(input_batch, num_datapoints, 2)  # [B, N, 2] in [0, 1] x [0, 1]
    elif dist == 'gaussian':
        # Should change to allow adjustable mean location in future.
        mean = torch.Tensor([.5, .5])[None, None, :]  # [1, 1, 2]
        inputs = torch.randn(input_batch, num_datapoints, 2) * gaussian_std + mean
    else:
        raise NotImplementedError

    inputs, outputs = get_input_output(inputs, images, target_channel,
                                        input_channels, continuous_output)

    return inputs, outputs  # [B, N, 2+], [B, N, 1]


def get_input_output(inputs, images, target_channel, input_channels,
                       continuous_output):
    """
    Args:
        inputs: [B, N, 2] in [0., 1.] x [0., 1.], input locations.
        images: [B, C, H, W] torch. NB: EXPECT VALUES TO BE IN [0, 1].
        target_channel: int, the index of the channel whose value we are
            targeting. For MNIST, this is just 0.
        input_channels: list of ints, indeces of the channels whose values
            are appended to the input locations x, y.
        continuous_output: bool, whether to provide a regression target instead
            of a classification one.

    Returns:
        inputs: [B, N, 2] torch if the images have a single channel. Else
            additional input channels are appended to the dimension, to get
            [B, N, 2 + num_additional_inputs] shape.
        outputs: [B, N, 1] torch. Assume that we only ever predict a single
            scalar output.
    """
    H_pixels = images.shape[-2]
    W_pixels = images.shape[-1]
    # Clamp in case the Gaussian distribution brings things out of bounds.
    # [B, N, 2] in [0., 1.] x [0., 1.]
    inputs = torch.clamp(inputs, min=1e-3, max=1. - 1e-3)

    # Quantise inputs
    coords = get_coords(inputs, H_pixels, W_pixels)  # [B, N, 2]

    # Get the target channel value corresponding to the inputs
    target_vals = index_into_image(images, coords, channel=target_channel)  # [B, N]

    if input_channels is not None:
        # Concatenate input channel value with x, y value
        for i in input_channels:
            input_channel_vals = index_into_image(images, coords, channel=i)  # [B, N]
            input_channel_vals = input_channel_vals[:, :, None]  # [B, N, 1]
            inputs = torch.cat([inputs, input_channel_vals], dim=2)  # [B, N, 2+]

    assert target_vals.max() <= 1. and target_vals.min() >= 0.
    if continuous_output:
        # Image values are in range [0, 1]
        outputs = target_vals[:, :, None]  # [B, N, output_dim=1]
    else:
        # Sample labels using the target channel value as a probability
        class_labels = 2. * torch.bernoulli(target_vals) - 1. # [B, N]
        outputs = class_labels[:, :, None]  # [B, N, output_dim=1]

    return inputs, outputs


class LambdaIterator:
    """Iterator that repeatedly generates elements from a lambda.

    Args:
        generator (function): Function that generates an element.
        num_elements (int): Number of elements to generate.
    """

    def __init__(self, generator, num_elements):
        self.generator = generator
        self.num_elements = num_elements
        self.index = 0

    def __next__(self):
        self.index += 1
        if self.index <= self.num_elements:
            return self.generator()
        else:
            raise StopIteration()

    def __iter__(self):
        return self


def climate_split(root, transform, num_train, num_val, batch_size, split):
    # Define function to load numpy files
    numpy_loader = lambda x: np.load(x)

    # Initialize dataset and return loader
    dataset = datasets.DatasetFolder(root=root,
                                     loader=numpy_loader,
                                     extensions='npy',
                                     transform=transform)

    # Perform train/val/test split.
    TOTAL_NUM = len(dataset)  # 15341 for full dataset, 731 for partial download.
    num_test = TOTAL_NUM - num_train - num_val
    assert num_test >= 0

    train_subset, val_subset, test_subset = random_split(dataset,
                                    [num_train, num_val, num_test],
                                    generator=torch.Generator().manual_seed(1))
    splits = {'train': [train_subset, num_train],
              'val': [val_subset, num_val],
              'test': [test_subset, num_test]}

    # For validation and test sets, batch size should be 1 and shuffle should be off?

    data_loader = DataLoader(dataset=splits[split][0],
                             shuffle=True,
                             batch_size=batch_size,
                             drop_last=True)
    data_iterator = iter(data_loader)
    
    # Number of batches in the dataset.
    # TODO: CHECK THIS.
    num_tasks = splits[split][1] // batch_size  # Rounded down since drop_last

    return data_loader, data_iterator, num_tasks


def MNIST_split(dataset, root, transform, num_train, num_val, batch_size, split):
    # Perform train/val split.
    train_set = dataset(root=root,
                        train=True,
                        download=True,
                        transform=transform)

    num_unused = 60000 - num_train - num_val  # Assume 60k datapoints
    train_subset, val_subset, _ = random_split(train_set,
                                   [num_train, num_val, num_unused],
                                   generator=torch.Generator().manual_seed(1))

    train_loader = DataLoader(dataset=train_subset,
                              shuffle=True,
                              batch_size=batch_size,
                              drop_last=True)
    val_loader = DataLoader(dataset=val_subset,
                            shuffle=True,
                            batch_size=batch_size,
                            drop_last=True)

    if split == 'train':
        data_loader = train_loader
        data_iterator = iter(train_loader)
        num_tasks = num_train // batch_size  # TODO: SHOULD THIS BE DIVIDED?
    elif split == 'val':
        data_loader = val_loader
        data_iterator = iter(val_loader)
        num_tasks = num_val // batch_size  # TODO: SHOULD THIS BE DIVIDED?
    elif split == 'test':
        test_set = dataset(root=root,
                           train=False,
                           download=True,
                           transform=transform)
        test_loader = DataLoader(dataset=test_set,
                                 shuffle=True,
                                 batch_size=batch_size,
                                 drop_last=True)
        data_loader = test_loader
        data_iterator = iter(test_loader)
        num_tasks = 10000 // batch_size  # TODO: SHOULD THIS BE DIVIDED?
    else:
        raise NotImplementedError

    return data_loader, data_iterator, num_tasks


def compute_slope_bias(mean, std, num_sigmas=2):
    """Compute slope and bias necessary to transform the range
    [mu - num_sigmas * sigma, mu + num_sigmas * sigma] gets mapped to [0, 1]
    using x' = slope * x + bias."""
    slope = 1 / (2 * num_sigmas * std)
    bias = -mean / (2 * num_sigmas * std) + 0.5
    return slope, bias


class AffineTransform():
    """Perform a channel-wise affine transform, with a different slope and bias
    parameter for each channel."""
    def __init__(self, slopes, biases):
        """
        Args:
            slopes: list of floats. The c'th float is the slope for the c'th
                channel transform.
            biases: list of floats. The c'th bias is the bias for the c'th
                channel transform.
        """
        assert len(slopes) == len(biases)
        self.slopes = slopes
        self.biases = biases

    def __call__(self, images):
        """
        Args:
            images: [C, H, W] torch, images
        """
        num_channels = len(self.slopes)
        assert num_channels == images.shape[0]
        for i in range(num_channels):
            images[i, :, :] = self.slopes[i] * images[i, :, :] + self.biases[i]
        return images


class AffineSubtractionTransform():
    """Subtract an affine transform of one channel from another channel."""
    def __init__(self, slope, bias, predictor_channel, target_channel):
        """
        Args:
            slope:
            bias:
            predictor_channel:
            target_channel:
        """
        self.slope = slope
        self.bias = bias
        self.predictor_channel = predictor_channel
        self.target_channel = target_channel

    def __call__(self, images):
        """
        Args:
            images: [C, H, W] torch, images
        """
        predictor_channel = images[self.predictor_channel, :, :]  # [H, W]
        prediction = predictor_channel * self.slope + self.bias  # [H, W]
        images[self.target_channel, :, :] = \
            images[self.target_channel, :, :] - prediction
        return images


class ClampTransform():
    """Clamp values in image, with a different clamping range for each
    channel."""
    def __init__(self, clamp_ranges):
        """
        Args:
            clamp_ranges: list of tuples. The c'th tuple is
                (MIN_VALUE, MAX_VALUE) for the c'th channel.
        """
        self.clamp_ranges = clamp_ranges

    def __call__(self, images):
        """
        Args:
            images: [C, H, W] torch, images
        """
        num_channels = len(self.clamp_ranges)
        assert num_channels == images.shape[0]
        for i in range(num_channels):
            min_max = self.clamp_ranges[i]  # Tuple of (MIN_VALUE, MAX_VALUE)
            images[i, :, :] = torch.clamp(images[i, :, :],
                                          min=min_max[0], max=min_max[1])
        return images


class ImageGenerator():

    def __init__(
            self,
            dataset='mnist',
            batch_size=20,
            train_points=50,
            test_points=50,
            pixels_per_dim=60,
            sampling='gaussian',
            gaussian_std=.2,
            num_train=50000,
            num_val=256,
            split='train',
            continuous_output=False
    ):
        self.batch_size = batch_size
        self.train_points = train_points
        self.test_points = test_points
        self.pixels_per_dim = pixels_per_dim  # For resizing MNIST and fMNIST
        self.sampling = sampling
        self.gaussian_std = gaussian_std
        self.split = split  # Whether to use train, test or val set.
        self.continuous_output = continuous_output

        if dataset == 'mnist':
            self.dataset = datasets.MNIST
            self.root = 'mnist/data'
            self.target_channel = 0
            self.input_channels = None
        elif dataset == 'fmnist':
            self.dataset = datasets.FashionMNIST
            self.root = 'fmnist/data'
            self.target_channel = 0
            self.input_channels = None
        elif dataset == 'climate':
            self.root = 'climate/data'
            self.target_channel = 1  # Temperature as target.
            self.input_channels = [0]  # Take orography as a side-input.
        else:
            raise NotImplementedError

        # Transform data and perform train/val/test split.
        if dataset == 'climate':
            # Use orography as linear predictor for temperature and subtract.
            affine_subtraction_transform = \
                self.compute_affine_subtraction_transform(prev_transform=transforms.ToTensor())
            unnorm_transform = transforms.Compose([
                transforms.ToTensor(),
                affine_subtraction_transform,
            ])

            # Bring everything into range [0., 1.]
            affine_01_transform, clamp_01_transform = \
                self.compute_range_01_transform(prev_transform=unnorm_transform,
                                                num_sigmas=2)

            total_transform = transforms.Compose([
                transforms.ToTensor(),
                affine_subtraction_transform,
                affine_01_transform,
                clamp_01_transform
            ])

            self.data_loader, self.data_iterator, self.num_tasks \
                = climate_split(self.root, total_transform, num_train, num_val,
                                batch_size, split)
        else:
            # Increase resolution and blur the image for MNIST and fMNIST
            blur_kernel_size = (pixels_per_dim // 4) * 2 + 1
            transform = transforms.Compose([
                transforms.Resize(size=pixels_per_dim),
                transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=.5),
                transforms.ToTensor()
            ])

            self.data_loader, self.data_iterator, self.num_tasks \
                = MNIST_split(self.dataset, self.root, transform,
                              num_train, num_val, batch_size, split)

    def cat_images_in_loader(self, data_loader):
        for iter, images in enumerate(data_loader):
            images, _ = images  # Labels not used.
            if iter == 0:
                all_images = images
            else:
                all_images = torch.cat([all_images, images], dim=0)  # [total_B, C, H, W]
        return all_images

    def compute_affine_subtraction_transform(self, prev_transform):
        data_loader, data_iterator, num_tasks \
            = climate_split(self.root, prev_transform,
                            num_train=500, num_val=0,
                            batch_size=20, split='train')

        all_images = self.cat_images_in_loader(data_loader)

        # Use orography (channel 0) as a linear predictor for temperature (channel 1)
        oro = all_images[:, 0, :, :]  # [B, H, W]
        temp = all_images[:, 1, :, :]  # [B, H, W]

        oro_mean = oro.mean()
        temp_mean = temp.mean()
        oro_dev = oro - oro_mean
        temp_dev = temp - temp_mean

        # OLS solution. beta is slope, alpha is bias
        beta = (oro_dev * temp_dev).sum() / (oro_dev ** 2).sum()
        alpha = temp_mean - (beta * oro_mean)
        self.norm_slope = beta
        self.norm_bias = alpha
        affine_subtraction_transform = AffineSubtractionTransform(
                                                    slope=self.norm_slope,
                                                    bias=self.norm_bias,
                                                    predictor_channel=0,
                                                    target_channel=1)
        return affine_subtraction_transform

    def compute_range_01_transform(self, prev_transform, num_sigmas):
        """Compute transform that makes everything in range [0, 1]."""
        data_loader, data_iterator, num_tasks \
            = climate_split(self.root, prev_transform,
                            num_train=500, num_val=0,
                            batch_size=20, split='train')

        all_images = self.cat_images_in_loader(data_loader)

        # Compute mean and std.
        oro = all_images[:, 0, :, :]  # [B, H, W]
        t2m = all_images[:, 1, :, :]  # [B, H, W]
        tp = all_images[:, 2, :, :]  # [B, H, W]
        oro_mean = oro.mean()
        oro_std = oro.std()
        t2m_mean = t2m.mean()
        t2m_std = t2m.std()
        tp_mean = tp.mean()
        tp_std = tp.std()

        # Compute slope and bias needed to send to [0, 1]
        oro_slope, oro_bias = compute_slope_bias(oro_mean, oro_std, num_sigmas)
        t2m_slope, t2m_bias = compute_slope_bias(t2m_mean, t2m_std, num_sigmas)
        tp_slope, tp_bias = compute_slope_bias(tp_mean, tp_std, num_sigmas)
        slopes = [oro_slope, t2m_slope, tp_slope]
        biases = [oro_bias, t2m_bias, tp_bias]

        affine_01_transform = AffineTransform(slopes=slopes, biases=biases)

        # Bring everything in [0., 1.] to match MNIST ranges. This means
        # that observations more than 3 sigma from the mean get clipped.
        clamp_ranges = [(0., 1.), (0., 1.), (0., 1.)]
        clamp_01_transform = ClampTransform(clamp_ranges=clamp_ranges)

        # Report roughly what percentage of values end up clipped by the
        # clamp_01_transform.
        oro_fraction = self.fraction_in_range(oro, oro_slope, oro_bias)
        t2m_fraction = self.fraction_in_range(t2m, t2m_slope, t2m_bias)
        tp_fraction = self.fraction_in_range(tp, tp_slope, tp_bias)

        print(f"Fraction unclipped for oro: {oro_fraction}")
        print(f"Fraction unclipped for t2m: {t2m_fraction}")
        print(f"Fraction unclipped for tp: {tp_fraction}")

        return affine_01_transform, clamp_01_transform

    def fraction_in_range(self, values, slope, bias):
        affine_values = values * slope + bias
        in_range = torch.logical_and(affine_values >= 0., affine_values <= 1.)
        fraction = in_range.sum() / in_range.numel()
        return fraction

    def get_next_batch(self):
        # Get a single batch of images.
        try:
            images, labels = next(self.data_iterator)  # [B, C, H, W], [B]
        except StopIteration:
            # Init a new iterator
            self.data_iterator = iter(self.data_loader)
            images, labels = next(self.data_iterator)
        return images, labels  # [B, C, H, W], [B]

    def generate_task(self, forced_input_locs=None):
        """Load a task, which is a batch of datasets, where each dataset is
        based on an MNIST image.
        Args:
            forced_input_locs: [B, N, 2] torch. If not None, the input locations
                are forced to these values. This is helpful for obtaining the
                orography on the grid.

        Returns:
            dict: A task, which is a dictionary with keys `x`, `y`, `x_context`,
                `y_context`, `x_target`, and `y_target.
        """

        # Get a single batch of images.
        images, _ = self.get_next_batch()  # [B, C, H, W], [B]
        num_points = self.train_points + self.test_points

        # [B, N_total, 2], [B, N_total, 1]
        x, y = sample_from_image(images,
                                 batch_size=self.batch_size,
                                 num_datapoints=num_points,
                                 dist=self.sampling,
                                 gaussian_std=self.gaussian_std,
                                 continuous_output=self.continuous_output,
                                 target_channel=self.target_channel,
                                 input_channels=self.input_channels,
                                 forced_input_locs=forced_input_locs)

        # Split into train and test set
        out = {"x": x,  # [B, N_total, 2+]
               "y": y,  # [B, N_total, 1]
               "x_context": x[:, :self.train_points, :],  # [B, N_train, 2]
               "y_context": y[:, :self.train_points, :],  # [B, N_train, 1]
               "x_target": x[:, self.train_points:, :],  # [B, N_test, 2]
               "y_target": y[:, self.train_points:, :],  # [B, N_test, 1]
               "image": images}  # [B, P, P]
        return {k: v.to(device) for k, v in out.items()}

    def __iter__(self):
        return LambdaIterator(lambda: self.generate_task(), self.num_tasks)

class DataGenerator(metaclass=abc.ABCMeta):
    """Data generator for GP samples.

    Args:
        batch_size (int, optional): Batch size. Defaults to 16.
        num_tasks (int, optional): Number of tasks to generate per epoch.
            Defaults to 256.
        x_range (tuple[float], optional): Range of the inputs. Defaults to
            [-2, 2].
        max_train_points (int, optional): Number of training points. Must be at
            least 3. Defaults to 50.
        max_test_points (int, optional): Number of testing points. Must be at
            least 3. Defaults to 50.
    """

    def __init__(
        self,
        batch_size=16,
        num_tasks=256,
        x_range=(-2, 2),
        max_train_points=50,
        max_test_points=50,
        input_dim=1,
    ):
        self.batch_size = batch_size
        self.num_tasks = num_tasks
        self.x_range = x_range
        self.max_train_points = max(max_train_points, 3)
        self.max_test_points = max(max_test_points, 3)
        self.input_dim = input_dim

        self.f_path = None  # Path of `self.f`
        self.f = None  # File to load data from

    @abc.abstractmethod
    def sample(self, x):
        """Sample at inputs `x`.

        Args:
            x (vector): Inputs to sample at.

        Returns:
            vector: Sample at inputs `x`.
        """

    def generate_dataset(self):
        """Generate just a single dataset."""
        # Determine number of test and train points.
        num_train_points = self.max_train_points
        num_test_points = self.max_test_points
        # num_train_points = np.random.randint(3, self.max_train_points + 1)
        # num_test_points = np.random.randint(3, self.max_test_points + 1)
        num_points = num_train_points + num_test_points

        # Sample inputs and outputs.
        if self.input_dim == 1:
            shape = [num_points]
        elif self.input_dim == 2:
            shape = [num_points, 2]
        else:
            raise NotImplementedError
        x = _rand(self.x_range, *shape)
        y = self.sample(x)

        # Determine indices for train and test set.
        inds = np.random.permutation(x.shape[0])

        if self.input_dim == 1:  # Sort.
            inds_train = sorted(inds[:num_train_points])
            inds_test = sorted(inds[num_train_points:num_points])

            return {"x": sorted(x), "y": y[np.argsort(x)],
                    "x_context": x[inds_train], "y_context": y[inds_train],
                    "x_target": x[inds_test], "y_target": y[inds_test]}
        elif self.input_dim == 2:  # Unsorted.
            return {"x": x,  # [N_total, 2]
                    "y": y,  # [N_total]
                    "x_context": x[:num_train_points, :],  # [N_train, 2]
                    "y_context": y[:num_train_points],  # [N_train]
                    "x_target": x[num_train_points:, :],  # [N_test, 2]
                    "y_target": y[num_train_points:]}  # [N_test]

    def generate_task(self):
        """Generate a task, which is a batch of datasets.

        Returns:
            dict: A task, which is a dictionary with keys `x`, `y`, `x_context`,
                `y_context`, `x_target`, and `y_target.
        """
        task = {
            "x": [],
            "y": [],
            "x_context": [],
            "y_context": [],
            "x_target": [],
            "y_target": [],
        }

        for _ in range(self.batch_size):
            dataset = self.generate_dataset()
            # Record to task.
            for key in task.keys():
                task[key].append(dataset[key])

        # Stack batch and convert to PyTorch.
        task = {
            k: torch.tensor(_uprank(np.stack(v, axis=0)), dtype=torch.float32).to(
                device
            )
            for k, v in task.items()
        }

        return task

    def __iter__(self):
        return LambdaIterator(lambda: self.generate_task(), self.num_tasks)

    @staticmethod
    def _get_path(key):
        caches_dir = os.path.join(Path(__file__).parent.absolute(), "_data_caches")
        # Ensure that the directory to save the caches in exists.
        os.makedirs(caches_dir, exist_ok=True)
        return os.path.join(caches_dir, f"{key}.pickle")

    def save(self, key, num_tasks):
        """Save generator to a file.
        Args:
            key (str): Name of generator to save.
            num_tasks (int): Number of batches of tasks to save.
        """
        with open(DataGenerator._get_path(key), "wb") as f:
            pickle.dump(self.batch_size, f)
            pickle.dump(self.num_tasks, f)
            pickle.dump(self.x_range, f)
            pickle.dump(self.max_train_points, f)
            pickle.dump(self.max_test_points, f)

            num_datasets = num_tasks * self.batch_size
            datasets = [self.generate_dataset() for _ in range(num_datasets)]
            pickle.dump(datasets, f)

    @classmethod
    def load(cls, key):
        """Load a saved generator.
        Args:
            key (str): Name of saved generator.
        Yields:
            dict: Task.
        """
        return SavedGenerator(key)


class SavedGenerator(DataGenerator):
    """A saved generator.
    Args:
        key (str): Name of the saved generator.
    """

    def __init__(self, key, input_dim=1):
        self.f = open(DataGenerator._get_path(key), "rb")
        batch_size = pickle.load(self.f)
        num_tasks = pickle.load(self.f)
        x_range = pickle.load(self.f)
        max_train_points = pickle.load(self.f)
        max_test_points = pickle.load(self.f)
        self.datasets = pickle.load(self.f)  # Load all the tasks
        self.batch_count = 0

        DataGenerator.__init__(
            self,
            batch_size,
            num_tasks,
            x_range,
            max_train_points,
            max_test_points,
            input_dim
        )

    def sample(self, x):
        raise RuntimeError("Cannot sample from a saved generator.")

    def generate_task(self):
        """Load a task, which is a batch of datasets. When an epoch is finished,
        shuffles the dataset.

        Returns:
            dict: A task, which is a dictionary with keys `x`, `y`, `x_context`,
                `y_context`, `x_target`, and `y_target.
        """
        task = {
            "x": [],
            "y": [],
            "x_context": [],
            "y_context": [],
            "x_target": [],
            "y_target": [],
        }

        if self.batch_count >= self.num_tasks:
            # Shuffle the tasks at the end of an epoch
            shuffle(self.datasets)
            self.batch_count = 0

        for i in range(self.batch_size):
            dataset = self.datasets[self.batch_count * self.batch_size + i]
            # Record to task.
            for key in task.keys():
                task[key].append(dataset[key])

        # Stack batch and convert to PyTorch.
        task = {
            k: torch.tensor(_uprank(np.stack(v, axis=0)), dtype=torch.float32).to(
                device
            )
            for k, v in task.items()
        }

        self.batch_count += 1

        return task


class GPGenerator(DataGenerator):
    """Generate samples from a GP with a given kernel.

    Further takes in keyword arguments for :class:`.data.DataGenerator`.

    Args:
        kernel (:class:`stheno.Kernel`, optional): Kernel to sample from.
            Defaults to an EQ kernel.
    """

    def __init__(self, kernel=stheno.EQ(), **kw_args):
        self.gp = stheno.GP(kernel)
        DataGenerator.__init__(self, **kw_args)

    def sample(self, x):
        return np.squeeze(self.gp(x).sample())


class ClassificationGPGenerator(GPGenerator):
    """Generate classification samples from a GP with a given kernel.
    The class labels are formed by taking the sign of the GP sample.

    Args:
        kernel (:class:`stheno.Kernel`, optional): Kernel to sample from.
            Defaults to an EQ kernel.
    """

    def __init__(self, kernel=stheno.EQ(), noisy=False, **kw_args):
        self.gp = stheno.GP(kernel)
        self.noisy = noisy  # Add aleatoric noise to the class labels.
        DataGenerator.__init__(self, **kw_args)

    def sample(self, x):
        if self.noisy:
            logits = np.squeeze(self.gp(x).sample())
            probs = _sigmoid(logits)
            return 2. * bernoulli.rvs(probs) - 1.
        else:
            return np.sign(np.squeeze(self.gp(x).sample()))


class RegressionGPGenerator(ClassificationGPGenerator):
    """Generate squashed regression samples from the GP, which live in range
    [0, 1].
    Args:
        kernel (:class:`stheno.Kernel`, optional): Kernel to sample from.
            Defaults to an EQ kernel.
    """

    def __init__(self, **kw_args):
        super().__init__(**kw_args)

    def sample(self, x):
        return 1 / (1 + np.exp(-self.gp(x).sample()))


class BalancedClassificationGPGenerator(GPGenerator):
    """Like ClassificationGP Generator, however, performs rejection sampling
    to ensure that the classes are approximately balanced.

    NB: rejection sampling is expected to be quite inefficient.

     Args:
        kernel (:class:`stheno.Kernel`, optional): Kernel to sample from.
            Defaults to an EQ kernel.
    """

    def __init__(self, kernel=stheno.EQ(), noisy=False, **kw_args):
        self.gp = stheno.GP(kernel)
        self.noisy = noisy  # Add aleatoric noise to the class labels.
        # Number of points used to check class balance
        self.num_balance_check = 300
        self.balance_threshold = 0.05
        DataGenerator.__init__(self, **kw_args)

    def sample(self, x):
        x_size = len(x)
        x_balance_check = np.linspace(self.x_range[0],
                                      self.x_range[1],
                                      self.num_balance_check)  # [N_balance]
        x_all = np.concatenate((x, x_balance_check))  # [N + N_balance]
        reject_batch_size = 100

        while True: # Rejection sampling for balanced tasks.
            gp_samples = self.gp(x_all).sample(reject_batch_size)
            # gp_samples = np.squeeze(np.sign(gp_samples))  # [N + N_balance, B]
            gp_samples = np.squeeze(gp_samples)  # [N + N_balance, B]
            for i in range(reject_batch_size):
                candidate_sample = gp_samples[:, i]  # [N + N_balance]
                balance_points = candidate_sample[x_size:]  # [N_balance]
                if self.balanced(balance_points):
                    if self.noisy:
                        probs = _sigmoid(candidate_sample[:x_size])
                        return 2. * bernoulli.rvs(probs) - 1.
                    else:
                        return np.sign(candidate_sample[:x_size])  # [N]

    def balanced(self, gp_sample):
        """Check if the sample is approximately balanced between classes"""
        if self.noisy:
            probs = _sigmoid(gp_sample)  # [N_balance]
            avg_prob = np.mean(probs)
        else:
            class_sample = np.sign(gp_sample)
            avg_prob = np.mean((class_sample + 1) / 2)
        return np.abs(avg_prob - 0.5) <= self.balance_threshold
