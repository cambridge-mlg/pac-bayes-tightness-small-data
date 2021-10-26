import numpy as np
import random
import argparse
import json
import torch
from pacbayes.experiment import WorkingDirectory
from pacbayes.args_utils import load_model
from pacbayes.utils import set_all_seeds
from pacbayes.evaluation import eval_and_save
from pacbayes.args_utils import setup_generators


def save_metrics(names, eval_lists):
    # Compute means and standard errors and place in dict.
    metrics = dict()
    for i, name in enumerate(names):
        num_datasets = len(eval_lists[i])
        mean = np.mean(eval_lists[i])
        se = np.std(eval_lists[i]) / np.sqrt(num_datasets)
        mean, se = mean.item(), se.item()

        key_mean = name + '_mean'
        key_se = name + '_std_error'
        metrics[key_mean] = mean
        metrics[key_se] = se
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Post-meta-training optimisation args
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
    parser.add_argument('--test_set',
                        type=str,
                        help='Name of file containing saved val data.')

    # Experiment args
    parser.add_argument('--root',
                        type=str,
                        help='Experiment root. Evaluation results will be '
                             'saved here.')
    args = parser.parse_args()

    # Set all random seeds.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Load working directory.
    if args.root:
        wd = WorkingDirectory(root=args.root)
    else:
        raise ValueError('You must specify a root directory to evaluate.')

    # Load the previous argparse arguments.
    args_file = wd.file('commandline_args.txt')
    with open(args_file, 'r') as f:
        prev_args = parser.parse_args()
        prev_args.__dict__ = json.load(f)

    # Load model using previous args.
    model = load_model(prev_args)

    # Load saved model.
    load_dict = torch.load(wd.file('checkpoint.pth.tar', exists=True))
    model.load_state_dict(load_dict['state_dict'])

    # Edit the previous args to include new post optimisation args.
    vars(prev_args)['post_iters'] = args.post_iters
    vars(prev_args)['post_learning_rate'] = args.post_learning_rate
    if args.test_set:  # If a test set is explicitly given during test time.
        vars(prev_args)['test_set'] = args.test_set

    # Save metrics.
    set_all_seeds(0)
    gen, gen_test = setup_generators(prev_args)
    eval_and_save(prev_args, wd, gen_test, model, post_optimise=False)

    set_all_seeds(0)
    gen, gen_test = setup_generators(prev_args)
    eval_and_save(prev_args, wd, gen_test, model, post_optimise=True)
