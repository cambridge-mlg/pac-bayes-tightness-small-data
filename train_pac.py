import argparse
import json
import torch
import time
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

from pacbayes.experiment import (
    report_loss,
    generate_root,
    WorkingDirectory,
    save_checkpoint
)
from pacbayes.plotting import batch_eval_dataset
from pacbayes.args_utils import setup_generators, load_model
from pacbayes.dists import Dist
from pacbayes.utils import _to_numpy, set_all_seeds
from pacbayes.evaluation import eval_and_save
from pacbayes.networks import set_grad


def validate(data, model, report_freq):
    """Compute the validation loss."""
    model.eval()
    losses = list()
    with torch.no_grad():
        for step, task in enumerate(data):
            loss_dict = model.loss(task['x_context'], task['y_context'])
            avg_risk_bound = loss_dict['risk_bounds']  # [B]
            avg_risk_bound = avg_risk_bound.mean().cpu().detach().item()
            losses.append(avg_risk_bound)
            avg_loss = np.array(losses).mean()
            report_loss('Validation', avg_loss, step, report_freq)
    avg_loss = np.array(losses).mean()
    return avg_loss


def train(data, model, opt, report_freq, tb_writer, global_step):
    """Perform a training epoch."""
    model.train()
    losses = list()
    start = time.time()
    for step, task in enumerate(data):
        loss_dict = model.loss(task['x_context'], task['y_context'])
        loss = loss_dict['loss']
        loss.backward()
        opt.step()
        opt.zero_grad()

        # Track training progress in terms of average risk bound
        avg_risk_bound = loss_dict['risk_bounds']  # [B]
        avg_risk_bound = avg_risk_bound.mean().cpu().detach().item()
        losses.append(avg_risk_bound)
        avg_loss = np.array(losses).mean()
        report_loss('Training', avg_loss, step, report_freq)
        terms_to_tb(tb_writer, loss_dict, global_step=global_step)
        global_step += 1
    print(f"Time for one epoch: {time.time() - start}")
    return avg_loss, global_step


def terms_to_tb(writer: SummaryWriter, terms: dict, global_step: int) -> None:
    for key, val in terms.items():
        if isinstance(val, torch.Tensor):
            val = _to_numpy(val.mean())
            writer.add_scalar(key, val, global_step=global_step)
        elif isinstance(val, Dist):
            for param_name, param in val.params.items():
                avg = _to_numpy(param.mean())
                writer.add_scalar(key + '_avg_' + param_name, avg,
                                  global_step=global_step)
        else:
            writer.add_scalar(key, val, global_step=global_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument('--data',
                        choices=['eq',
                                 'matern',
                                 'noisy-mixture',
                                 'weakly-periodic',
                                 'mnist',
                                 'fmnist',
                                 'climate'],
                        default='eq',
                        help='Data set to train the CNP on. ')
    parser.add_argument('--input_dim',
                        type=int,
                        default=1,
                        help='Dimension of GP inputs.')
    parser.add_argument('--num_threads',
                        type=int,
                        help='Number of threads.')
    parser.add_argument('--class_scheme',
                        choices=['standard',
                                 'balanced'],
                        default='standard',
                        help='Whether to use standard or balanced '
                             'data generator. ')
    parser.add_argument('--continuous_output',
                        action='store_true',
                        help='Whether to use continuous outputs for MNIST.')
    parser.add_argument('--noisy',
                        action='store_true',
                        help='Whether to add aleatoric noise to GP samples.')
    parser.add_argument('--stretch',
                        type=float,
                        default=0.7,
                        help='Stretch factor of lengthscale of kernel.')
    parser.add_argument('--scale',
                        type=float,
                        default=1.,
                        help='Scale of kernel.')
    parser.add_argument('--num_context',
                        type=int,
                        default=30,
                        help='Number of points to use in context set at train '
                             'and test time')
    parser.add_argument('--num_test',
                        type=int,
                        default=300,
                        help='Number of points used to estimate generalization'
                             ' risk')
    parser.add_argument('--train_set',
                        type=str,
                        help='Name of file containing saved train data.')
    parser.add_argument('--test_set',
                        type=str,
                        help='Name of file containing saved val data.')
    parser.add_argument('--num_train_batches',
                        type=int,
                        default=5000,
                        help='Number of batches of tasks per epoch for train'
                             ' for 1D classification, or number of images per'
                             ' epoch for 2D image classification.')
    parser.add_argument('--num_test_batches',
                        type=int,
                        default=16,
                        help='Number of batches of tasks per epoch for test'
                             ' for 1D classification, or number of images per'
                             ' epoch for 2D image classification.')

    # Image sampling args
    parser.add_argument('--input_dist',
                        choices=['uniform',
                                 'gaussian'],
                        default='gaussian',
                        help='Input distribution for MNIST.')
    parser.add_argument('--input_std',
                        default=0.2,
                        type=float,
                        help='Input standard deviation for MNIST, if Gaussian'
                             ' sampling is used.')

    # Meta-training args
    parser.add_argument('--epochs',
                        default=100,
                        type=int,
                        help='Number of epochs to train for.')
    parser.add_argument('--learning_rate',
                        default=1e-3,
                        type=float,
                        help='Learning rate.')
    parser.add_argument('--weight_decay',
                        default=1e-5,
                        type=float,
                        help='Weight decay.')
    parser.add_argument('--reg_epochs',
                        default=0,
                        type=int,
                        help='Number of epochs to train regression loss for.')
    parser.add_argument('--frozen_beta_epochs',
                        default=0,
                        type=int,
                        help='Number of epochs to freeze catoni beta for.')
    parser.add_argument('--frozen_beta',
                        default=3.,
                        type=float,
                        help='Value to freeze catoni beta at.')
    parser.add_argument('--frozen_feature_map_epochs',
                        default=0,
                        type=int,
                        help='Number of epochs to feature map for.')
 
    # Bound args
    parser.add_argument('--model',
                        choices=['catoni',
                                 'catoni-ddp',
                                 'catoni-amortised-beta',
                                 'maurer',
                                 'maurer-ddp',
                                 'maurer-inv',
                                 'maurer-inv-ddp',
                                 'maurer-optimistic',
                                 'maurer-optimistic-ddp',
                                 'maurer-inv-optimistic',
                                 'maurer-inv-optimistic-ddp',
                                 'convex-separable',
                                 'convex-separable-ddp',
                                 'convex-nonseparable',
                                 'convex-nonseparable-ddp',
                                 'hoeff-val',
                                 'kl-val'],
                        default='catoni',
                        help='Model/PAC bound to use.')
    parser.add_argument('--delta',
                        default=0.1,
                        type=float,
                        help='Delta probability of the PAC bound not holding.')
    parser.add_argument('--prior_proportion',
                        default=0.25,
                        type=float,
                        help='Proportion of data to use in a sample dependent'
                             'prior/proportion of data to use in train set for '
                             'validation models.')

    # Post-meta-training optimisation args
    parser.add_argument('--post_optimise',
                        action='store_true',
                        help='If set, optimise the posterior after '
                             'meta-training, during the training run.')
    parser.add_argument('--post_iters',
                        default=3000,
                        type=int,
                        help='Number of iterations to optimise post '
                             'meta-training.')
    parser.add_argument('--post_learning_rate',
                        default=3e-4,
                        type=float,
                        help='Learning rate for post-meta-training '
                             'optimisation.')

    # DeepSets architecture args
    parser.add_argument('--feature_map',
                        choices=['mlp',
                                 'image_features'],
                        default='mlp',
                        help='Type of feature map to use (when not using '
                             'ConvDeepSets).')
    parser.add_argument('--init_pca',
                        action='store_true',
                        help='Init image features to top PCA components of'
                             'MNIST.')
    parser.add_argument('--feature_dim',
                        type=int,
                        default=256,
                        help='Number of features in the feature map defining'
                             'the predictor function')
    parser.add_argument('--feature_layers',
                        type=int,
                        default=2,
                        help='Number of hidden layers in the feature map')
    parser.add_argument('--feature_width',
                        type=int,
                        default=512,
                        help='Width of hidden layers in the feature map')
    parser.add_argument('--rep_dim',
                        type=int,
                        default=512,
                        help='Dimension of the deep sets representation of the'
                             'context set')
    parser.add_argument('--enc_layers',
                        type=int,
                        default=2,
                        help='Number of hidden layers in the deep sets '
                             'encoder.')
    parser.add_argument('--dec_layers',
                        type=int,
                        default=2,
                        help='Number of hidden layers in the deep sets '
                             'decoder.')
    parser.add_argument('--enc_width',
                        type=int,
                        default=512,
                        help='Width of hidden layers in the deep sets '
                             'encoder.')
    parser.add_argument('--dec_width',
                        type=int,
                        default=512,
                        help='Width of hidden layers in the deep sets '
                             'decoder.')
    parser.add_argument('--dist_family',
                        choices=['mean-field',
                                 'full-cov'],
                        default='mean-field',
                        help='Prior/posterior family to use.')

    # ConvDeepSets architecture args
    parser.add_argument('--conv',
                        action='store_true',
                        help='Use convolutional model.')
    parser.add_argument('--cnn_type',
                        choices=['simple',
                                 'unet',
                                 'simple_separable',
                                 'simple_xl',
                                 'hourglass'],
                        default='simple',
                        help='CNN type.')
    parser.add_argument('--cnn_channels',
                        type=int,
                        default=16,
                        help='Number of CNN channels.')
    parser.add_argument('--points_per_unit',
                        type=int,
                        default=32,
                        help='Number of points per unit for convolutional'
                             'model.')
    parser.add_argument('--internal_multiplier',
                        type=int,
                        default=2,
                        help='Factor by which points per unit is increased '
                             'internally relative to the feature dimension.')

    # Experiment args
    parser.add_argument('--load',
                        action='store_true',
                        help='Load a model from the experiment root. If this'
                             'is not specified, a new model is initialised.')
    parser.add_argument('--root',
                        type=str,
                        help='Experiment root, which is the directory from '
                             'which the experiment will run. If it is not '
                             'given, a directory will be automatically '
                             'created.')
    parser.add_argument('--load_args',
                        action='store_true',
                        help='Load the args from a json file in the experiment'
                             'root.')

    args = parser.parse_args()
    args.val_proportion = 1. - args.prior_proportion

    # Set number of threads.
    if args.num_threads:
        torch.set_num_threads(args.num_threads)

    # Set all random seeds.
    set_all_seeds(0)

    # Load working directory.
    if args.root:
        wd = WorkingDirectory(root=args.root)
    else:
        experiment_name = f'{args.data}'
        task = 'mnist' if args.data == 'mnist' else '1D'
        wd = WorkingDirectory(root=generate_root(experiment_name, task))

    # Save argparse arguments.
    args_file = wd.file('commandline_args.txt')
    with open(args_file, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Load argparse arguments.
    to_load = args.load
    if args.load_args:
        with open(args_file, 'r') as f:
            args.__dict__ = json.load(f)
    # If the original run was not loaded, and a subsequent run wants to use
    # the args of the original run, we should still load the model.
    args.load = to_load

    # Setup generators.
    gen, gen_test = setup_generators(args)

    # Load model.
    model = load_model(args)

    # Define optimiser.
    opt = Adam(model.parameters(),
               args.learning_rate,
               weight_decay=args.weight_decay)

    prev_epochs = 0  # Number of epochs the loaded model trained for.
    if args.load:
        # Attempt to load saved model and optimiser.
        load_dict = torch.load(wd.file('checkpoint.pth.tar', exists=True))
        model.load_state_dict(load_dict['state_dict'])
        opt.load_state_dict(load_dict['optimizer'])
        prev_epochs = load_dict['epoch']

    # Writer for tensorboard logging, log in subdirectory "logs"
    writer = SummaryWriter(log_dir=f'{wd.root}/logs')

    # Perform training.
    best_loss = np.inf
    global_step = 0
    for epoch in range(prev_epochs, prev_epochs + args.epochs):
        print(f'\nEpoch: {epoch + 1}/{prev_epochs + args.epochs}')

        if epoch < args.reg_epochs:
            # Start by using NLL loss for the sake of initialisation.
            model.loss_fn = 'nll'
        elif args.continuous_output:
            # Squared error loss, and squashed regression model.
            model.loss_fn = 'regression'
        else:
            # 0-1 classification loss.
            model.loss_fn = 'classification'

        # Freeze Catoni beta initially.
        if epoch < args.frozen_beta_epochs:
            model.set_beta(freeze_beta=True, freeze_beta_val=args.frozen_beta)
        elif hasattr(model, 'set_beta'):
            model.set_beta(freeze_beta=False)

        # Freeze feature map initially.
        if epoch < args.frozen_feature_map_epochs:
            set_grad(model.architecture.feature_map, requires_grad=False)
        else:
            set_grad(model.architecture.feature_map, requires_grad=True)

        # Compute training loss.
        train_loss, global_step = train(gen, model, opt, report_freq=50,
                                   tb_writer=writer, global_step=global_step)
        report_loss('Training', train_loss, 'epoch')

        # Compute validation loss.
        val_loss = validate(gen_test, model, report_freq=50)
        report_loss('Validation', val_loss, 'epoch')

        # Write validation loss to tensorboard.
        val_dict = {'held_out_risk_bounds': val_loss}
        terms_to_tb(writer, val_dict, global_step=global_step)

        if epoch % 3 == 0:
            # Get a batch of tasks.
            task = gen_test.generate_task()

            # Plot first datasets in the batch.
            for i in range(1):
                x, y = task['x_context'][i:i+1], task['y_context'][i:i+1]
                x_test, y_test = task['x_target'][i:i+1], task['y_target'][i:i+1]

                # For plotting MNIST ground truth image.
                image = task['image'][i:i+1] if args.data in \
                                    ['mnist', 'fmnist', 'climate'] else None
                if args.data in ['mnist', 'fmnist']:
                    target_channel = 0  # Only one channel.
                elif args.data == 'climate':
                    target_channel = 1  # Temperature is second channel.
                else:
                    target_channel = None

                _ = batch_eval_dataset(model, x, y, x_test, y_test,
                                       image=image,
                                       plot=True,
                                       epoch=epoch,
                                       wd=wd,
                                       optimise=args.post_optimise,
                                       iters=args.post_iters,
                                       learning_rate=args.post_learning_rate,
                                       target_channel=target_channel,
                                       data_gen=gen_test)

        # Update the best objective value and checkpoint the model.
        is_best = False
        if val_loss < best_loss:
            best_obj = val_loss
            is_best = True
        save_checkpoint(wd,
                        {'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_loss': best_loss,
                         'optimizer': opt.state_dict()},
                        is_best=is_best)

    # Compute and save metrics with and without post optimisation.
    # Reset data gens in between to share seed.
    set_all_seeds(0)
    gen, gen_test = setup_generators(args)
    eval_and_save(args, wd, gen_test, model, post_optimise=False)

    set_all_seeds(0)
    gen, gen_test = setup_generators(args)
    eval_and_save(args, wd, gen_test, model, post_optimise=True)
