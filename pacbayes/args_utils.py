import stheno.torch as stheno
from .models import (
    CatoniClassifier,
    CatoniDDPClassifier,
    CatoniAmortisedBetaClassifier,
    MaurerClassifier,
    MaurerDDPClassifier,
    MaurerInvClassifier,
    MaurerInvDDPClassifier,
    ConvexClassifier,
    ConvexDDPClassifier,
    HoeffValClassifier,
    KLValClassifier
)
from .architectures import (
    DeepSetsArchitecture,
    DeepSetsPriorArchitecture,
    DeepSetsPriorAmortisedBeta,
    ConvDeepSetsArchitecture,
    ConvDeepSetsPriorArchitecture,
    GNPArchitecture,
    GNPPriorArchitecture
)
from .data import (
    ClassificationGPGenerator,
    BalancedClassificationGPGenerator,
    RegressionGPGenerator,
    ImageGenerator
)
from .networks import (
    SimpleConv,
    UNet,
    MLPFeatureMap,
    ImageFeaturesMap,
    SimpleConv2d,
    SimpleSeparableConv2d,
    HourGlassConv2d,
    SimpleConv2dXL
)
from .dists import MeanFieldGaussianFamily, FullCovGaussianFamily
from .utils import device


FIXED_PRIOR_MODELS = ['catoni', 'maurer', 'maurer-optimistic', 'maurer-inv',
                      'maurer-inv-optimistic', 'convex-separable',
                      'convex-nonseparable', 'hoeff-val', 'kl-val']

DDP_MODELS_NO_AMORTISED_BETA = ['catoni-ddp', 'maurer-ddp', 'maurer-optimistic-ddp',
              'maurer-inv-ddp', 'maurer-inv-optimistic-ddp',
              'convex-separable-ddp', 'convex-nonseparable-ddp']

DDP_MODELS_AMORTISED_BETA = ['catoni-amortised-beta']


def load_image_data_gen(args, num_context, num_test, split='train'):

    gen = ImageGenerator(dataset=args.data,
                         continuous_output=args.continuous_output,
                         batch_size=20,
                         train_points=num_context,
                         test_points=num_test,
                         pixels_per_dim=60,
                         sampling=args.input_dist,
                         gaussian_std=args.input_std,
                         num_train=args.num_train_batches,
                         num_val=args.num_test_batches,
                         split=split)

    return gen


def setup_generators(args):
    # Setup generators.
    if args.data in ['mnist', 'fmnist', 'climate']:
        gen = load_image_data_gen(args,
                                  num_context=args.num_context,
                                  num_test=args.num_test,
                                  split='train')
        gen_test = load_image_data_gen(args,
                                       num_context=args.num_context,
                                       num_test=args.num_test,
                                       split='val')
    else:
        gen = load_data_gen(args,
                            num_batches=args.num_train_batches,
                            max_train_points=args.num_context,
                            max_test_points=0)
        if args.train_set:  # Load pre-generated data.
            gen = gen.load(args.train_set)

        gen_test = load_data_gen(args,
                                 num_batches=args.num_test_batches,
                                 max_train_points=args.num_context,
                                 max_test_points=args.num_test)
        if args.test_set:  # Load pre-generated data.
            gen_test = gen_test.load(args.test_set)
    return gen, gen_test


def load_data_gen(args, num_batches, max_train_points, max_test_points):

    if args.data == 'eq':
        kernel = args.scale * stheno.EQ().stretch(args.stretch)
    elif args.data == 'matern':
        kernel = args.scale * stheno.Matern52().stretch(args.stretch)
    elif args.data == 'noisy-mixture':
        kernel = stheno.EQ().stretch(args.stretch) + \
                 stheno.EQ().stretch(args.stretch * 0.25) + \
                 1e-3 * stheno.Delta()
        kernel = args.scale * kernel
    elif args.data == 'weakly-periodic':
        kernel = stheno.EQ().stretch(0.5 * args.scale) * \
                 stheno.EQ().periodic(period=0.25 * args.scale)
        kernel = args.scale * kernel
    else:
        raise ValueError(f'Unknown data "{args.data}".')

    if args.continuous_output:  # Squashed regression.
        GenClass = RegressionGPGenerator
    else: # Classification.
        # Choose classification data generation scheme.
        if args.class_scheme == 'standard':
            GenClass = ClassificationGPGenerator
        elif args.class_scheme == 'balanced':
            GenClass = BalancedClassificationGPGenerator
        else:
            raise ValueError(f'Unknown class scheme "{args.class_scheme}".')

    if args.input_dim == 1:
        x_range = (-2., 2.)
    elif args.input_dim == 2:
        x_range = (0., 1.)
    else:
        raise NotImplementedError

    gen = GenClass(kernel=kernel,
                   noisy=args.noisy,  # Noisy does nothing for regression.
                   x_range=x_range,
                   batch_size=16,
                   num_tasks=num_batches,
                   max_train_points=max_train_points,
                   max_test_points=max_test_points,
                   input_dim=args.input_dim)

    return gen


def load_model(args):
    # Initialise the distribution family.
    if args.dist_family == 'mean-field':
        dist_family = MeanFieldGaussianFamily()
    elif args.dist_family == 'full-cov':
        dist_family = FullCovGaussianFamily()
    else:
        raise NotImplementedError

    # Handle old args dictionaries that didn't bother to specify some newer args.
    if not hasattr(args, 'input_dim'):
        vars(args)['input_dim'] = 1
    if not hasattr(args, 'feature_map'):
        vars(args)['feature_map'] = 'mlp'
    if not hasattr(args, 'continuous_output'):
        vars(args)['continuous_output'] = False

    if args.data == 'climate':
        input_dim = 3  # Since orography is appended as another input.
    elif args.data in ['mnist', 'fmnist'] or args.input_dim == 2:
        input_dim = 2
    elif args.input_dim == 1:
        input_dim = 1
    else:
        raise NotImplementedError

    # Initialise the architecture.
    if args.conv:
        if input_dim == 2:
            x_lims = (-0.1, 1.1)
            if args.cnn_type == 'simple':
                cnn = SimpleConv2d()
            elif args.cnn_type == 'simple_separable':
                cnn = SimpleSeparableConv2d()
            elif args.cnn_type == 'simple_xl':
                cnn = SimpleConv2dXL()
            elif args.cnn_type == 'hourglass':
                cnn = HourGlassConv2d()
            else:
                raise NotImplementedError
        else:  # 1D.
            x_lims = (-2.1, 2.1)
            # Initialise CNN.
            if args.cnn_type == 'simple':
                cnn = SimpleConv(channels=args.cnn_channels)
            elif args.cnn_type == 'unet':
                cnn = UNet(channels=args.cnn_channels)
            else:
                raise NotImplementedError

        if args.dist_family == 'mean-field':  # ConvCNP
            # Initialise architecture.
            if args.model in FIXED_PRIOR_MODELS:
                ArchitectureClass = ConvDeepSetsArchitecture
            elif args.model in DDP_MODELS_NO_AMORTISED_BETA:
                ArchitectureClass = ConvDeepSetsPriorArchitecture
            else:
                raise NotImplementedError
            architecture = ArchitectureClass(input_dims=input_dim,
                                             x_lims=x_lims,
                                             points_per_unit=args.points_per_unit,
                                             cnn=cnn,
                                             dist_family=dist_family)
        elif args.dist_family == 'full-cov':  # GNP
            # Initialise architecture.
            if args.model in FIXED_PRIOR_MODELS:
                ArchitectureClass = GNPArchitecture
            elif args.model in DDP_MODELS_NO_AMORTISED_BETA:
                ArchitectureClass = GNPPriorArchitecture
            else:
                raise NotImplementedError
            architecture = ArchitectureClass(x_lims=x_lims,
                                 points_per_unit=args.points_per_unit,
                                 internal_multiplier=args.internal_multiplier)
        else:
            raise NotImplementedError
    else:  # Use MLPs
        # Initialise architecture.
        if args.model in FIXED_PRIOR_MODELS:
            ArchitectureClass = DeepSetsArchitecture
        elif args.model in DDP_MODELS_NO_AMORTISED_BETA:
            ArchitectureClass = DeepSetsPriorArchitecture
        elif args.model in DDP_MODELS_AMORTISED_BETA:
            ArchitectureClass = DeepSetsPriorAmortisedBeta
        else:
            raise NotImplementedError

        # Initialise feature map.
        if args.feature_map == 'image_features':
            feature_map = ImageFeaturesMap(feature_dim=args.feature_dim,
                                           init_PCA=args.init_pca)
        elif args.feature_map == 'mlp':
            feature_map = MLPFeatureMap(input_dim=input_dim,
                              feature_dim=args.feature_dim,
                              num_layers=args.feature_layers,
                              width=args.feature_width)
        else:
            raise NotImplementedError

        architecture = ArchitectureClass(input_dim=input_dim,
                                         output_dim=1,
                                         feature_map=feature_map,
                                         rep_dim=args.rep_dim,
                                         enc_layers=args.enc_layers,
                                         enc_width=args.enc_width,
                                         dec_layers=args.dec_layers,
                                         dec_width=args.dec_width,
                                         dist_family=dist_family)

    # Initialise the model.
    if args.model == 'catoni':
        model = CatoniClassifier(beta=1.,  # initialisation value
                                 architecture=architecture,
                                 delta=args.delta)
    elif args.model == 'catoni-ddp':
        model = CatoniDDPClassifier(beta=1.,  # initialisation value
                                    architecture=architecture,
                                    delta=args.delta,
                                    prior_proportion=args.prior_proportion)
    elif args.model == 'catoni-amortised-beta':
        model = CatoniAmortisedBetaClassifier(
                                      architecture=architecture,
                                      prior_proportion=args.prior_proportion,
                                      delta=args.delta)
    elif args.model == 'maurer':
        model = MaurerClassifier(architecture=architecture, delta=args.delta)
    elif args.model == 'maurer-ddp':
        model = MaurerDDPClassifier(architecture=architecture,
                                    delta=args.delta,
                                    prior_proportion=args.prior_proportion)
    elif args.model == 'maurer-inv':
        model = MaurerInvClassifier(architecture=architecture,
                                    delta=args.delta,
                                    optimistic=False)
    elif args.model == 'maurer-inv-ddp':
        model = MaurerInvDDPClassifier(architecture=architecture,
                                       delta=args.delta,
                                       prior_proportion=args.prior_proportion,
                                       optimistic=False)
    elif args.model == 'maurer-optimistic':
        model = MaurerClassifier(architecture=architecture,
                                 delta=args.delta,
                                 optimistic=True)
    elif args.model == 'maurer-optimistic-ddp':
        model = MaurerDDPClassifier(architecture=architecture,
                                    delta=args.delta,
                                    prior_proportion=args.prior_proportion,
                                    optimistic=True)
    elif args.model == 'maurer-inv-optimistic':
        model = MaurerInvClassifier(architecture=architecture,
                                    delta=args.delta,
                                    optimistic=True)
    elif args.model == 'maurer-inv-optimistic-ddp':
        model = MaurerInvDDPClassifier(architecture=architecture,
                                       delta=args.delta,
                                       prior_proportion=args.prior_proportion,
                                       optimistic=True)
    elif args.model == 'convex-separable':
        model = ConvexClassifier(architecture=architecture,
                                 delta=args.delta,
                                 separable=True)
    elif args.model == 'convex-separable-ddp':
        model = ConvexDDPClassifier(architecture=architecture,
                                    delta=args.delta,
                                    prior_proportion=args.prior_proportion,
                                    separable=True)
    elif args.model == 'convex-nonseparable':
        model = ConvexClassifier(architecture=architecture,
                                 delta=args.delta,
                                 separable=False)
    elif args.model == 'convex-nonseparable-ddp':
        model = ConvexDDPClassifier(architecture=architecture,
                                    delta=args.delta,
                                    prior_proportion=args.prior_proportion,
                                    separable=False)
    elif args.model == 'hoeff-val':
        model = HoeffValClassifier(architecture=architecture,
                                   val_proportion=args.val_proportion)
    elif args.model == 'kl-val':
        model = KLValClassifier(architecture=architecture,
                                val_proportion=args.val_proportion)
    else:
        raise ValueError(f'Unknown model "{args.model}"')

    if args.continuous_output:
        # Squared error loss, and squashed regression model.
        model.loss_fn = 'regression'
    else:
        # 0-1 classification loss.
        model.loss_fn = 'classification'

    model.to(device)
    return model
