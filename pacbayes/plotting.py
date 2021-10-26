import torch
import numpy as np
import matplotlib.pyplot as plt
from wbml.plot import tweak
from .classifiers import PACBayesDDPClassifier, ValClassifier
from .models import (
    ConvexClassifier,
    ConvexDDPClassifier,
    KLValClassifier,
)
from .utils import device, _to_numpy, _list_to_numpy_squeeze


def to_cpu(*tensors):
    return [tensor.cpu() if tensor is not None else None for tensor in tensors]


def make_fig_axes(num_axes):
    width = num_axes * 2.7
    height = 4
    fig, axes = plt.subplots(1, num_axes, figsize=(width, height))
    return fig, axes


def transform_class_prob(prob):
    """Convert class probs from [0, 1] to [-1, 1], to match the image values."""
    return 2. * prob - 1.


def transform_range(y):
    """Convert range of y from {-1, +1} to {0, 1}"""
    return (y + 1.) / 2.


def trivial_risk(y, y_test):
    """
    Args:
        y: [B, N, 1], torch
        y_test: [B, N, 1], torch

    Returns:
        trivial_risk: [B], numpy
    """
    # Compute trivial generalisation risk, obtained by always predicting the
    # more prevalent class on the training set.
    y = y.detach().cpu().numpy()[:, :, 0]  # [B, N]
    y_test = y_test.detach().cpu().numpy()[:, :, 0]  # [B, N]

    y_sum = np.sum(y, axis=1)  # [B]
    prevalent_class = np.sign(y_sum + 1e-8)  # [B], jitter to break ties
    correct = np.maximum(y_test * prevalent_class[:, None], 0.) # [B, N_test]
    trivial_risk = 1. - np.mean(correct, axis=-1)  # [B]
    return trivial_risk


def trivial_reg_risk(y, y_test):
    """
    N.B. this assumes y values for regression are always in [0, 1].
    Args:
        y: [B, N, 1], torch
        y_test: [B, N, 1], torch

    Returns:
        trivial_risk: [B], numpy
    """
    # Compute the trivial generalisation risk of a regression model that
    # always predicts the mean value.
    y = y.detach().cpu().numpy()[:, :, 0]  # [B, N]
    y_test = y_test.detach().cpu().numpy()[:, :, 0]  # [B, N]

    y_mean = np.mean(y, axis=-1)  # [B]

    # y values are in range [0, 1].
    trivial_risk = np.mean((y_mean[:, None] - y_test) ** 2, axis=-1)  # [B]
    return trivial_risk


def add_scatter_plot(ax, x, y, c, title, marker="o", edgecolor=None, vmin=-1.,
                     vmax=0., climate=False):
    # vmin and vmax set in accordance with c being in [-1., 0.], which occurs
    # by taking the negative sign of the image value. This is so that high
    # image values get assigned lighter colours, so the MNIST has white ink.
    ax.scatter(x=x, y=y, c=c, cmap='Greys', vmin=vmin, vmax=vmax,
               marker=marker, s=30, edgecolors=edgecolor)
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_facecolor('blue')
    if climate:
        ax.set_aspect(aspect=121 / 221)
    else:
        ax.set_aspect('equal', 'box')
    ax.set_title(title)


def add_BW_image_plot(ax, image, title):
    ax.imshow(-image[0, :, :], cmap="Greys")
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(title)


def add_regression_mean_std_plot(ax1, ax2, x_grid, sigmoid_means, sigmoid_vars,
                                 name='Pred', climate=False):
    # Plot regression means and std after sigmoid.
    MAX_STD = .2
    add_scatter_plot(ax1, x=x_grid[0, :, 0], y=x_grid[0, :, 1],
                     c=1. - sigmoid_means,
                     vmin=0. , vmax=1. , title=f"{name} mean", climate=climate)
    grid_pred_std = MAX_STD - np.sqrt(sigmoid_vars)  # Minus sign so high uncertainty is white
    add_scatter_plot(ax2, x=x_grid[0, :, 0], y=x_grid[0, :, 1], c=grid_pred_std,
                     title=f"{name} std", vmin=0., vmax=MAX_STD, climate=climate)


def add_combined_dataset_plot(ax, x_set1, x_set2, y_set1, y_set2, title="",
                              climate=False):
    N1, N2 = x_set1.shape[1], x_set2.shape[1]
    x_all = torch.cat([x_set1[0, :, 0], x_set2[0, :, 0]])
    y_all = torch.cat([x_set1[0, :, 1], x_set2[0, :, 1]])
    c_all = torch.cat([-y_set1[0, :, 0], -y_set2[0, :, 0]])
    add_scatter_plot(ax, x=x_all, y=y_all,
                     c=c_all,
                     title=f"{title}: {N1 + N2}",
                     marker="o",
                     climate=climate)


def plot_dataset(x, y, x_test, y_test, x_grid, class_probs, **kwargs):
    """Make a classifier plot for a single dataset. x_context is [N] etc."""
    data = [x, y, x_test, y_test, x_grid]
    x, y, x_test, y_test, x_grid = _list_to_numpy_squeeze(data)
    y, y_test = transform_range(y), transform_range(y_test)
    N_context = x.shape[0]
    N_target = x_test.shape[0]

    fig, ax = plt.subplots()
    ax.set_xlim(-2., 2.)
    ax.set_ylim(-0.1, 1.1)

    # Plot NP predictions.
    ax.plot(x_grid, class_probs,
            label='Probability of prediction +1')

    # Context and target sets.
    size = 75
    # ax.scatter(x_test, y_test, label=f'Test set',
    #            color='black', marker='x', alpha=0.07, s=size)
    ax.scatter(x, y, label=f'Risk set, size {N_context}',
               color='C1', marker='x', s=size)


def plot_dataset_reg(x, y, x_test, y_test, x_grid, sigmoid_means,
                     sigmoid_vars, **kwargs):
    """Make a 1D regression plot for a single dataset. Shapes are all
    [1, N, input/output_dim]."""
    x, y, x_test, y_test, x_grid = to_cpu(x, y, x_test, y_test, x_grid)
    N_context = x.shape[1]
    N_target = x_test.shape[1]

    fig, ax = plt.subplots()
    ax.set_xlim(-2., 2.)
    ax.set_ylim(-0.1, 1.1)

    # Plot sigmoid mean.
    ax.plot(x_grid.squeeze(), sigmoid_means, label='Predictive mean')
    sigmoid_stds = np.sqrt(sigmoid_vars)
    ax.fill_between(x_grid.squeeze(),
                    sigmoid_means - sigmoid_stds,
                    sigmoid_means + sigmoid_stds, alpha=0.3)

    # Context and target sets.
    size = 75
    # ax.scatter(x_test, y_test, label=f'Test set',
    #            color='black', marker='x', alpha=0.07, s=size)
    ax.scatter(x, y, label=f'Risk set, size {N_context}',
               color='C1', marker='x', s=size)


def plot_dataset_DDP(model, x, y, x_test, y_test, x_grid, prior_class_probs,
                     post_class_probs, **kwargs):
    """Make a classifier plot for a single dataset when there is a data
    dependent prior. x_context is [N] etc."""
    input_list = model.data_split(x, y)  # [1, N, 1]
    x_prior, y_prior, x_risk, y_risk = _list_to_numpy_squeeze(input_list)
    x_test, y_test, x_grid = _list_to_numpy_squeeze([x_test, y_test, x_grid])
    y_prior, y_risk = transform_range(y_prior), transform_range(y_risk)
    y_test = transform_range(y_test)

    N_target = x_test.shape[0]
    N_prior = x_prior.shape[0]
    N_risk = x_risk.shape[0]

    fig, ax = plt.subplots()
    ax.set_xlim(-2., 2.)
    ax.set_ylim(-0.1, 1.1)

    # Predictions.
    ax.plot(x_grid, prior_class_probs, label='Prior')
    ax.plot(x_grid, post_class_probs, label='Posterior')

    # Context and target sets along with prior and risk sets.
    size = 75
    # ax.scatter(x_test, y_test, label=f'Test set',
    #            color='black', marker='x', alpha=0.07, s=size)
    ax.scatter(x_prior, y_prior, label=f'Prior set, size {N_prior}',
               color='tab:blue', marker='x', s=size)
    ax.scatter(x_risk, y_risk, label=f'Risk set, size {N_risk}',
               color='tab:orange', marker='x', s=size)


def plot_dataset_DDP_reg(model, x, y, x_test, y_test, x_grid,
                         prior_sigmoid_means, prior_sigmoid_vars,
                         sigmoid_means, sigmoid_vars, **kwargs):
    """Make a regression plot for a single dataset when there is a data
    dependent prior. x_context is [N] etc."""
    x, y, x_test, y_test, x_grid = to_cpu(x, y, x_test, y_test, x_grid)
    # [1, N, input/output_dim]
    x_prior, y_prior, x_risk, y_risk = model.data_split(x, y)

    N_prior = x_prior.shape[1]
    N_risk = x_risk.shape[1]

    fig, ax = plt.subplots()
    ax.set_xlim(-2., 2.)
    ax.set_ylim(-0.1, 1.1)

    # Predictions.
    ax.plot(x_grid.squeeze(), prior_sigmoid_means, label='Prior mean')
    prior_sigmoid_stds = np.sqrt(prior_sigmoid_vars)
    ax.fill_between(x_grid.squeeze(),
                    prior_sigmoid_means - prior_sigmoid_stds,
                    prior_sigmoid_means + prior_sigmoid_stds, alpha=0.3)

    ax.plot(x_grid.squeeze(), sigmoid_means, label='Posterior mean')
    sigmoid_stds = np.sqrt(sigmoid_vars)
    ax.fill_between(x_grid.squeeze(),
                    sigmoid_means - sigmoid_stds,
                    sigmoid_means + sigmoid_stds, alpha=0.3)

    # Context and target sets along with prior and risk sets.
    size = 75
    # ax.scatter(x_test, y_test, label=f'Test set',
    #            color='black', marker='x', alpha=0.07, s=size)
    ax.scatter(x_prior, y_prior, label=f'Prior set, size {N_prior}',
               color='tab:blue', marker='x', s=size)
    ax.scatter(x_risk, y_risk, label=f'Risk set, size {N_risk}',
               color='tab:orange', marker='x', s=size)


def plot_dataset_val(model, x, y, x_test, y_test, x_grid, class_probs,
                     **kwargs):
    """Make a classifier plot for a single dataset for the validation based
    model. x_context is [N] etc."""
    input_list = model.data_split(x, y)  # [1, N, 1]
    x_train, y_train, x_val, y_val = _list_to_numpy_squeeze(input_list)
    x_test, y_test, x_grid = _list_to_numpy_squeeze([x_test, y_test, x_grid])
    y_train, y_val = transform_range(y_train), transform_range(y_val)
    y_test = transform_range(y_test)

    N_target = x_test.shape[0]
    N_train = x_train.shape[0]
    N_val = x_val.shape[0]

    fig, ax = plt.subplots()
    ax.set_xlim(-2., 2.)
    ax.set_ylim(-0.1, 1.1)

    # Predictions.
    ax.plot(x_grid, class_probs,
             label='Prediction')

    # Context and target sets along with train and val sets.
    size = 75
    # ax.scatter(x_test, y_test, label=f'Test set',
    #             color='black', marker='x', alpha=0.07, s=size)
    ax.scatter(x_train, y_train, label=f'Train set, size {N_train}',
                color='tab:blue', marker='x', s=size)
    ax.scatter(x_val, y_val, label=f'Val set, size {N_val}',
                color='tab:orange', marker='x', s=size)


def plot_dataset_val_reg(model, x, y, x_test, y_test, x_grid,
                         sigmoid_means, sigmoid_vars, **kwargs):
    """Make a 1D regression plot for a single dataset for the validation based
    model. x_context is [N] etc."""
    x, y, x_test, y_test, x_grid = to_cpu(x, y, x_test, y_test, x_grid)
    x_train, y_train, x_val, y_val = model.data_split(x, y)  # [1, N, 1]

    N_train = x_train.shape[1]
    N_val = x_val.shape[1]

    fig, ax = plt.subplots()
    ax.set_xlim(-2., 2.)
    ax.set_ylim(-0.1, 1.1)

    # Plot sigmoid mean.
    ax.plot(x_grid.squeeze(), sigmoid_means, label='Predictive mean')
    sigmoid_stds = np.sqrt(sigmoid_vars)
    ax.fill_between(x_grid.squeeze(),
                    sigmoid_means - sigmoid_stds,
                    sigmoid_means + sigmoid_stds, alpha=0.3)

    # Context and target sets along with train and val sets.
    size = 75
    # ax.scatter(x_test, y_test, label=f'Test set',
    #            color='black', marker='x', alpha=0.07, s=size)
    ax.scatter(x_train, y_train, label=f'Train set, size {N_train}',
               color='tab:blue', marker='x', s=size)
    ax.scatter(x_val, y_val, label=f'Val set, size {N_val}',
               color='tab:orange', marker='x', s=size)


def plot_dataset_val_2D(model, x, y, x_test, y_test, x_grid, class_probs,
                        image, climate=False):
    """Make a 2D classifier plot for a single dataset for the validation based
    model. x_context is [N] etc."""
    class_probs = transform_class_prob(class_probs)
    x, y, x_test, y_test, x_grid, image = to_cpu(x, y, x_test, y_test, x_grid, image)
    x_train, y_train, x_val, y_val = model.data_split(x, y)  # [1, N, 1]

    N_target = x_test.shape[1]
    N_train = x_train.shape[1]

    num_axes = 4 if image is None else 5
    fig, axes = make_fig_axes(num_axes)

    # Add negative sign to colour map so that "ink" is white
    add_scatter_plot(axes[0], x=x_train[0, :, 0], y=x_train[0, :, 1],
                     c=-y_train[0, :, 0], title=f"Train: {N_train}", marker="o",
                     climate=climate)

    add_combined_dataset_plot(axes[1], x_train, x_val, y_train, y_val,
                              title="Train and val", climate=climate)

    add_scatter_plot(axes[2], x=x_grid[0, :, 0], y=x_grid[0, :, 1], c=-class_probs,
                     title="Predictions", climate=climate)
    add_scatter_plot(axes[3], x=x_test[0, :, 0], y=x_test[0, :, 1],
                     c=-y_test[0, :, 0], title=f"Target, size {N_target}",
                     climate=climate)

    if image is not None:
        add_BW_image_plot(axes[4], image, title="Ground truth.")


def plot_dataset_val_2D_reg(model, x, y, x_test, y_test, x_grid,
                            sigmoid_means, sigmoid_vars, image,
                            climate=False):
    """Make a 2D classifier plot for a single dataset for the validation based
    model. x_context is [N] etc."""
    x, y, x_test, y_test, x_grid, image = to_cpu(x, y, x_test, y_test, x_grid, image)
    x_train, y_train, x_val, y_val = model.data_split(x, y)  # [1, N, 1]

    N_target = x_test.shape[1]
    N_train = x_train.shape[1]

    num_axes = 5 if image is None else 6
    fig, axes = make_fig_axes(num_axes)

    # Add negative sign to colour map so that "ink" is white
    add_scatter_plot(axes[0], x=x_train[0, :, 0], y=x_train[0, :, 1],
                     c=-y_train[0, :, 0], title=f"Train: {N_train}", marker="o",
                     climate=climate)

    add_combined_dataset_plot(axes[1], x_train, x_val, y_train, y_val,
                              title="Train and val", climate=climate)

    # Regression means and std after sigmoid.
    add_regression_mean_std_plot(axes[2], axes[3], x_grid,
                                 sigmoid_means, sigmoid_vars, climate=climate)

    add_scatter_plot(axes[4], x=x_test[0, :, 0], y=x_test[0, :, 1],
                     c=-y_test[0, :, 0], title=f"Target, size {N_target}",
                     climate=climate)

    if image is not None:
        add_BW_image_plot(axes[5], image, title="Ground truth.")


def plot_dataset_DDP_2D(model, x, y, x_test, y_test, x_grid, prior_class_probs,
                     post_class_probs, image=None, climate=False):
    """Make a 2D classifier plot for a single dataset when there is a data
    dependent prior. Shapes are all [1, N, input/output_dim]."""
    prior_class_probs = transform_class_prob(prior_class_probs)
    post_class_probs = transform_class_prob(post_class_probs)

    x, y, x_test, y_test, x_grid, image = to_cpu(x, y, x_test, y_test, x_grid, image)
    # [1, N, input/output_dim]
    x_prior, y_prior, x_risk, y_risk = model.data_split(x, y)

    N_target = x_test.shape[1]
    N_prior = x_prior.shape[1]

    num_axes = 5 if image is None else 6
    fig, axes = make_fig_axes(num_axes)

    # Add negative sign to colour map so that "ink" is white
    add_scatter_plot(axes[0], x=x_prior[0, :, 0], y=x_prior[0, :, 1],
                     c=-y_prior[0, :, 0], title=f"Prior: {N_prior}", marker="o",
                     climate=climate)

    add_combined_dataset_plot(axes[1], x_prior, x_risk, y_prior, y_risk,
                              title="Prior and risk", climate=climate)

    add_scatter_plot(axes[2], x=x_grid[0, :, 0], y=x_grid[0, :, 1],
                     c=-prior_class_probs, title="Prior predictions", climate=climate)
    add_scatter_plot(axes[3], x=x_grid[0, :, 0], y=x_grid[0, :, 1],
                     c=-post_class_probs, title="Post. predictions", climate=climate)
    add_scatter_plot(axes[4], x=x_test[0, :, 0], y=x_test[0, :, 1],
                     c=-y_test[0, :, 0], title=f"Target, size {N_target}", climate=climate)

    if image is not None:
        add_BW_image_plot(axes[5], image, title="Ground truth.")


def plot_dataset_DDP_2D_reg(model, x, y, x_test, y_test, x_grid,
                            prior_sigmoid_means, prior_sigmoid_vars,
                            sigmoid_means, sigmoid_vars, image, climate=False):
    """Make a 2D classifier plot for a single dataset when there is a data
    dependent prior. Shapes are all [1, N, input/output_dim]."""
    x, y, x_test, y_test, x_grid, image = \
        to_cpu(x, y, x_test, y_test, x_grid, image)
    # [1, N, input/output_dim]
    x_prior, y_prior, x_risk, y_risk = model.data_split(x, y)

    N_target = x_test.shape[1]
    N_prior = x_prior.shape[1]

    num_axes = 7 if image is None else 8
    fig, axes = make_fig_axes(num_axes)

    # Add negative sign to colour map so that "ink" is white
    add_scatter_plot(axes[0], x=x_prior[0, :, 0], y=x_prior[0, :, 1],
                     c=-y_prior[0, :, 0], title=f"Prior: {N_prior}", marker="o",
                     climate=climate)

    add_combined_dataset_plot(axes[1], x_prior, x_risk, y_prior, y_risk,
                              title="Prior and risk", climate=climate)

    # Regression prior means and std after sigmoid.
    add_regression_mean_std_plot(axes[2], axes[3], x_grid,
                         prior_sigmoid_means, prior_sigmoid_vars, name='Prior',
                         climate=climate)

    # Regression pred means and std after sigmoid.
    add_regression_mean_std_plot(axes[4], axes[5], x_grid,
                                 sigmoid_means, sigmoid_vars, climate=climate)

    add_scatter_plot(axes[6], x=x_test[0, :, 0], y=x_test[0, :, 1],
                     c=-y_test[0, :, 0], title=f"Target, size {N_target}",
                     climate=climate)

    if image is not None:
        add_BW_image_plot(axes[7], image, title="Ground truth.")


def plot_dataset_2D(x, y, x_test, y_test, x_grid, class_probs, image,
                    climate=False):
    """Make a 2D classifier plot for a single dataset. Shapes are all
    [1, N, input/output_dim]."""
    class_probs = transform_class_prob(class_probs)
    x, y, x_test, y_test, x_grid, image = to_cpu(x, y, x_test, y_test, x_grid, image)
    N_context = x.shape[1]
    N_target = x_test.shape[1]

    num_axes = 3 if image is None else 4
    fig, axes = make_fig_axes(num_axes)

    # Add negative sign to colour map so that "ink" is white
    add_scatter_plot(axes[0], x=x[0, :, 0], y=x[0, :, 1], c=-y[0, :, 0],
                     title=f"Context, size {N_context}", climate=climate)
    add_scatter_plot(axes[1], x=x_grid[0, :, 0], y=x_grid[0, :, 1], c=-class_probs,
                     title="Predictions", climate=climate)
    add_scatter_plot(axes[2], x=x_test[0, :, 0], y=x_test[0, :, 1],
                     c=-y_test[0, :, 0], title=f"Target, size {N_target}",
                     climate=climate)

    if image is not None:
        add_BW_image_plot(axes[3], image, title="Ground truth.")


def plot_dataset_2D_reg(x, y, x_test, y_test, x_grid,
                        sigmoid_means, sigmoid_vars, image, climate=False):
    """Make a 2D regression plot for a single dataset. Shapes are all
    [1, N, input/output_dim]."""
    x, y, x_test, y_test, x_grid, image = to_cpu(x, y, x_test, y_test, x_grid, image)
    N_context = x.shape[1]
    N_target = x_test.shape[1]

    num_axes = 4 if image is None else 5
    fig, axes = make_fig_axes(num_axes)

    # Add negative sign to colour map so that "ink" is white
    add_scatter_plot(axes[0], x=x[0, :, 0], y=x[0, :, 1], c=-y[0, :, 0],
                     title=f"Context, size {N_context}", climate=climate)

    # Regression means and std after sigmoid.
    add_regression_mean_std_plot(axes[1], axes[2],
                                 x_grid, sigmoid_means, sigmoid_vars, climate=climate)

    add_scatter_plot(axes[3], x=x_test[0, :, 0], y=x_test[0, :, 1],
                     c=-y_test[0, :, 0], title=f"Target, size {N_target}",
                     climate=climate)

    if image is not None:
        add_BW_image_plot(axes[4], image, title="Ground truth.")


def plot_convex(model, x, epoch, wd, figdir):
    # Plot the learned convex function
    # rs = torch.linspace(0., 1., 200).to(device)
    # convex_pop_risks = model.convex_pop_risk(rs)
    # convex_emp_risks = model.convex_emp_risk(rs)
    # rs = rs.detach().cpu().numpy()
    # plt.plot(rs, convex_pop_risks.detach().cpu().numpy(), label='Pop risks')
    # plt.plot(rs, convex_emp_risks.detach().cpu().numpy(), label='Emp risks')
    # plt.legend()
    # convex_figname = f'epoch_{epoch}_convex_function.png'
    # convex_figpath = wd.file(figdir, convex_figname)
    # plt.savefig(convex_figpath)
    # plt.close()

    # Plot the function on the RHS that we need to compute the sup of
    N_context = x.shape[1]
    log_supremum_r = _to_numpy(model.compute_supremum(N_context))
    plt.plot(log_supremum_r)
    plt.title(f'Log supremum: {np.max(log_supremum_r)}')
    sup_figname = f'epoch_{epoch}_sup.png'
    sup_figpath = wd.file(figdir, sup_figname)
    plt.savefig(sup_figpath)
    plt.close()


def generate_title(model, eval_dict):
    if isinstance(model, KLValClassifier):
        chernoff_gen_bound = eval_dict['bounds']
        bin_tail_gen_bound = eval_dict['binomial_tail_bounds']
        gen_risk = eval_dict['gen_risk']
        val_risk = eval_dict['val_risk']
        title = f'Bin. tail/Chernoff bound: ${bin_tail_gen_bound:.2f}$/${chernoff_gen_bound:.2f}$, ' \
                f'gen. risk: ${gen_risk:.3f}$, ' \
                f'test risk: ${val_risk:.2f}$.'
    else:
        gen_bound = eval_dict['bounds']
        gen_risk = eval_dict['gen_risk']
        KL = eval_dict['KL']
        title = f'Gen. bound: ${gen_bound:.2f}$, gen. risk: ${gen_risk:.3f}$, ' \
                f'KL: ${KL:.2f}$.'

    return title


def batch_eval_dataset(model, x, y, x_test, y_test, image=None, plot=False,
                       epoch=None, wd=None, optimise=False, figdir='figs',
                       target_channel=0, data_gen=None, **kwargs):
    """
    Args:
        model:
        x: [B, N, input_dim] torch, context inputs
        y: [B, N, 1] torch, context outputs
        x_test: [B, N_test, input_dim] torch, generalisation inputs
        y_test: [B, N_test, 1] torch, generalisation outputs
        image: [B, C, H, W] torch, ground truth image for plotting
        plot: bool, to save plot
        epoch: int, epoch number
        wd: WorkingDirectory object
        optimise: bool, to post optimise
        target_channel: int, index of channel that is targeted for prediction.
            This is 0 for MNIST.
        data_gen: DataGenerator object. Should be passed for climate to help
            get orography.
        **kwargs: Post-optimisation kwargs

    Returns:
        eval_dict: dict, containing emp_risk, gen_risk, bounds
    """
    batch_size = x.shape[0]
    input_dim = x.shape[-1]
    if input_dim == 1:
        num_grid = 300
        x_grid = torch.linspace(-2., 2., num_grid)[None, :, None]  # [1, N_grid, 1]
    elif input_dim == 2:
        num_grid = 40  # 40 per axis, so 1600 grid points in total
        grid = torch.linspace(1e-3, 1. - 1e-3, num_grid)  # [N_grid]
        grid_1, grid_2 = torch.meshgrid(grid, grid)  # [N_grid, N_grid]
        grid_12 = torch.stack([grid_1, grid_2], dim=2)  # [N_grid, N_grid, 2]
        x_grid = grid_12.view(-1, 2)  # [N_grid ** 2, 2]
        x_grid = x_grid[None, :, :]  # [1, N_grid ** 2, 2] in [0., 1.]^2
    elif input_dim == 3:  # Climate data.
        num_grid_1 = 121
        num_grid_2 = 221
        grid_1 = torch.linspace(1e-3, 1. - 1e-3, num_grid_1)  # [N_grid]
        grid_2 = torch.linspace(1e-3, 1. - 1e-3, num_grid_2)  # [N_grid]
        grid_1, grid_2 = torch.meshgrid(grid_1, grid_2)  # [N_grid, N_grid]
        grid_12 = torch.stack([grid_1, grid_2], dim=2)  # [N_grid, N_grid, 2]
        x_grid = grid_12.view(-1, 2)  # [N_grid ** 2, 2]
        x_grid = x_grid[None, :, :]  # [1, N_grid ** 2, 2] in [0., 1.]^2
    else:
        raise NotImplementedError
    x_grid = x_grid.to(device)  # [1, N_grid ** 2, 2]

    climate = False
    if input_dim == 3:  # Climate.
        # Get the whole input, including orography channel corresponding to x_grid.
        # N.B. RELIES ON OROGRAPHY CHANNEL BEING CONSTANT ACROSS ALL IMAGES.
        climate = True
        task = data_gen.generate_task(forced_input_locs=x_grid)
        x_grid = task['x'][0:1, :, :]  # [1, N_grid ** 2, 3]

    # Evaluate the model.
    eval_dict = model.evaluate(x, y, x_test, y_test, x_grid,
                               post_optimise=optimise, **kwargs)
    for k, v in eval_dict.items():  # shape [] except class_probs, [B, N_grid]
        eval_dict[k] = _to_numpy(v)

    # Pop the predictive on x_grid.
    class_probs = eval_dict.pop('class_probs')  # [B, N_grid]
    sigmoid_means = eval_dict.pop('sigmoid_means')
    sigmoid_vars = eval_dict.pop('sigmoid_vars')
    if isinstance(model, PACBayesDDPClassifier):
        prior_class_probs = eval_dict.pop('prior_class_probs')  # [B, N_grid]
        prior_sigmoid_means = eval_dict.pop('prior_sigmoid_means')
        prior_sigmoid_vars = eval_dict.pop('prior_sigmoid_vars')

    # Get metrics of interest.
    if model.loss_fn == 'classification':
        eval_dict['trivial_risk'] = trivial_risk(y, y_test)  # [B]
    else:
        eval_dict['trivial_risk'] = trivial_reg_risk(y, y_test)  # [B]

    if plot:
        for i in range(batch_size):
            x_i, y_i = x[i:i+1, :, :], y[i:i+1, :, :]
            x_test_i, y_test_i = x_test[i:i+1, :, :], y_test[i:i+1, :, :]
            data = [x_i, y_i, x_test_i, y_test_i]

            # Extract the target channel, [1, H, W]
            image_i = None if image is None else image[i, target_channel, :, :]

            if isinstance(model, PACBayesDDPClassifier):
                if model.loss_fn == 'classification':
                    plot_fun = plot_dataset_DDP if input_dim == 1 else plot_dataset_DDP_2D
                    plot_fun(model, *data, x_grid, prior_class_probs[i], class_probs[i], image=image_i)
                else: # Regression model
                    plot_fun = plot_dataset_DDP_reg if input_dim == 1 else plot_dataset_DDP_2D_reg
                    plot_fun(model, *data, x_grid, prior_sigmoid_means[i], prior_sigmoid_vars[i],
                             sigmoid_means[i], sigmoid_vars[i], image=image_i, climate=climate)
            elif isinstance(model, ValClassifier):
                if model.loss_fn == 'classification':
                    plot_fun = plot_dataset_val if input_dim == 1 else plot_dataset_val_2D
                    plot_fun(model, *data, x_grid, class_probs[i], image=image_i)
                else: # Regression model
                    plot_fun = plot_dataset_val_reg if input_dim == 1 else plot_dataset_val_2D_reg
                    plot_fun(model, *data, x_grid, sigmoid_means[i], sigmoid_vars[i],
                             image=image_i, climate=climate)
            else:  # PAC-Bayes model without data-dependent prior
                if model.loss_fn == 'classification':
                    plot_fun = plot_dataset if input_dim == 1 else plot_dataset_2D
                    plot_fun(*data, x_grid, class_probs[i], image=image_i)
                else: # Regression model
                    plot_fun = plot_dataset_reg if input_dim == 1 else plot_dataset_2D_reg
                    plot_fun(*data, x_grid, sigmoid_means[i], sigmoid_vars[i],
                             image=image_i, climate=climate)

            # Record metrics in plot title and save.
            eval_dict_i = {k: v[i] for k, v in eval_dict.items()}
            title = generate_title(model, eval_dict_i)
            # title = str()
            # for k, v in eval_dict.items():
            #     title += (k + f': {v:.3f}, ')

            if input_dim == 2 or input_dim == 3:
                plt.suptitle(title, fontsize=12)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            elif input_dim == 1:
                # plt.legend(fontsize=13)  # Only 1D classification uses legend.
                fig = plt.gcf()
                fig.set_size_inches(6, 2.5)

                ax = plt.gca()
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(11)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(11)
                plt.title(title, fontsize=12)

            tweak(legend=False, grid=True)

            figname = f'epoch_{epoch}_dataset_{i}.pdf'
            figpath = wd.file(figdir, figname)
            plt.savefig(figpath)
            plt.close()

            # if isinstance(model, ConvexClassifier) or \
            #         isinstance(model, ConvexDDPClassifier):
            #     plot_convex(model, x, epoch, wd, figdir)

    return eval_dict
