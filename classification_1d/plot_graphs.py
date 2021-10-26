import matplotlib.pyplot as plt
import itertools
import numpy as np
import json
import argparse
from pathlib import Path
from pacbayes.experiment import WorkingDirectory
from wbml.plot import tweak


def load_mean_std(wd, key):
    saved_args = wd.file('commandline_args.txt')
    with open(saved_args, 'r') as f:
        args_dict = json.load(f)
        ddp_models = ['catoni-ddp', 'catoni-amortised-beta',
                      'maurer-ddp', 'maurer-inv-ddp', 'convex-nonseparable-ddp',
                      'maurer-optimistic-ddp', 'maurer-inv-optimistic-ddp']
        fixed_prior_models = ['catoni', 'maurer', 'maurer-inv', 'convex-nonseparable',
                              'maurer-optimistic', 'maurer-inv-optimistic']

        if args_dict['model'] in ddp_models:
            prop = args_dict['prior_proportion']
        elif args_dict['model'] in ['kl-val', 'hoeff-val']:
            prop = args_dict['prior_proportion']  # New val model args rely entirely on prior proportion.
            # prop = 1. - args_dict['val_proportion']
        elif args_dict['model'] in fixed_prior_models:
            prop = 0.
        else:
            raise NotImplementedError

        lr = args_dict['learning_rate']
    
    if args.post_optimise:
        metrics_file = 'eval_metrics_post_opt.txt'
    else:
        metrics_file = 'eval_metrics_no_post_opt.txt'
    metrics = wd.file(metrics_file)
    with open(metrics, 'r') as f:
        metrics_dict = json.load(f)
        mean = metrics_dict[key + '_mean']
        std_error = metrics_dict[key + '_std_error']
    return prop, mean, std_error, lr


def make_plot_lists(dirs, key, plot_lr=False):
    xs, ys, es = list(), list(), list()
    for dir in dirs:
        wd = WorkingDirectory(root=dir)
        prop, mean, std_error, lr = load_mean_std(wd, key)
        if plot_lr:
            xs.append(lr)
        else:
            xs.append(prop)
        ys.append(mean)
        es.append(std_error)
    xs = np.array(xs)
    ys = np.array(ys)
    es = np.array(es)
    return xs, ys, es


def plot_graph(key, catoni_dirs, maurer_dirs, maurer_optimistic_dirs,
               catoni_optimistic_dirs, biconvex_dirs, val_dirs, figpath,
               title, plot_lr=False, legend=True):

    fig, ax = plt.subplots()
    fig.set_size_inches(3.5, 3.06)
    labelled_dirs = [(catoni_dirs, 'Catoni'),
                     (maurer_dirs, 'Maurer'),
                     (maurer_optimistic_dirs, 'Optimistic Maurer'),
                     (catoni_optimistic_dirs, 'Optimistic Catoni'),
                     (biconvex_dirs, 'Learned convex'),
                     (val_dirs, 'Chernoff test set'),
                     (val_dirs, 'Binomial tail test set')]
    # marker = itertools.cycle(('v', 's', 'o', 'd', '<', '>', '*'))

    # For the MLP plot.
    marker = itertools.cycle(('v', '<', '>'))
    colors = itertools.cycle(('tab:blue', 'tab:purple', 'tab:brown'))

    for labelled_dir in labelled_dirs:
        dirs, label = labelled_dir
        if dirs is not None:
            if label == 'Optimistic Catoni' and key == 'bound':
                eval_dict_key = 'optimistic_bounds'
            elif label == 'Binomial tail test set' and key == 'bound':
                eval_dict_key = 'binomial_tail_bounds'
            else:
                eval_dict_key = key
            xs, ys, es = make_plot_lists(dirs, eval_dict_key, plot_lr=plot_lr)
            xs, ys = zip(*sorted(zip(xs, ys)))  # Sort by x value
            current_color = next(colors)
            ax.errorbar(xs, ys, yerr=es, label=label, marker=next(marker),
                        color=current_color, # For the MLP plot.
                        markersize=10, markeredgecolor=None)
            ax.fill_between(xs, ys - 2 * es, ys + 2 * es, alpha=0.3,
                            color=current_color)  # For MLP plot

    ylims = {'emp_risk': (0., .025),
             'bound': (0.1, 0.25),
             'gen_risk': (0.025, .06)}
    # ylims = {'emp_risk': (0., .1),
    #          'bound': (0.2, 0.40),
    #          'gen_risk': (0.04, .15)}


    ax.set_ylim(ylims[key][0], ylims[key][1])
    ax.set_title(title)
    if plot_lr:
        ax.set_xlabel('Learning rate')
        ax.set_xscale('log')
    else:
        ax.set_xlabel('Prior/train proportion')
    ax.set_ylabel('Risk')
    ax.grid()

    locs, labels = plt.xticks()  # Get the current locations and labels.
    plt.xticks(np.arange(0, 1., step=0.2))  # Set label locations.

    tweak(legend=legend, legend_loc='best', grid=True)
    plt.tight_layout()
    plt.savefig(figpath, dpi=100)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c',
                        '--catoni_dirs',
                        nargs='*',
                        help='Catoni directories to plot.')
    parser.add_argument('-m',
                        '--maurer_dirs',
                        nargs='*',
                        help='Maurer directories to plot.')
    parser.add_argument('-o',
                        '--maurer_optimistic_dirs',
                        nargs='*',
                        help='Maurer optimistic directories to plot.')
    parser.add_argument('-co',
                        '--catoni_optimistic_dirs',
                        nargs='*',
                        help='Catoni optimistic directories to plot. Takes '
                        'the optimistic bound from a normal catoni run.')
    parser.add_argument('-b',
                        '--biconvex_dirs',
                        nargs='*',
                        help='Biconvex directories to plot.')
    parser.add_argument('-v',
                        '--val_dirs',
                        nargs='*',
                        help='Validation directories to plot.')
    parser.add_argument('--plot_dir',
                        type=str,
                        help='Directory to save plot.')
    parser.add_argument('--name',
                        type=str,
                        help='Name of plot.')
    parser.add_argument('--post_optimise',
                        action='store_true',
                        help='If true, plot post optimised metrics.')
    parser.add_argument('--plot_lr',
                        action='store_true',
                        help='If true, plot performance against learning rate')


    args = parser.parse_args()

    titles = {'emp_risk': 'Empirical Risk',
              'bound': 'Generalisation Bound',
              'gen_risk': 'Generalisation Risk'}

    for key in ['emp_risk', 'bound', 'gen_risk']:
        if args.post_optimise:
            post = 'post_opt'
        else:
            post = 'no_post_opt'
        name = args.name + '_' + key + '_' + post + '_legend.pdf'
        figpath = Path(args.plot_dir, name)
        plot_graph(key, args.catoni_dirs, args.maurer_dirs,
                   args.maurer_optimistic_dirs, args.catoni_optimistic_dirs,
                   args.biconvex_dirs, args.val_dirs, figpath,
                   title=titles[key], plot_lr=args.plot_lr, legend=True)

        name = args.name + '_' + key + '_' + post + '_no_legend.pdf'
        figpath = Path(args.plot_dir, name)
        plot_graph(key, args.catoni_dirs, args.maurer_dirs,
                   args.maurer_optimistic_dirs, args.catoni_optimistic_dirs,
                   args.biconvex_dirs, args.val_dirs, figpath,
                   title=titles[key], plot_lr=args.plot_lr, legend=False)
