import json
import numpy as np
import math
from pacbayes.plotting import batch_eval_dataset


def extract_metric(metric_name, eval_dict):
    if metric_name in eval_dict:
        metric = eval_dict[metric_name]
    else:  # Validation models will not have an optimistic bound.
        metric = np.array([math.nan])
    return metric


def append_metric(metric, metric_name, eval_dict):
    if metric_name in eval_dict:
        metric = np.concatenate([metric, eval_dict[metric_name]])
    else:  # Validation models will not have an optimistic bound.
        metric = np.array([math.nan])
    return metric


def eval(args, wd, gen_test, model, post_optimise):
    """Evaluate the model after training."""
    model.eval()

    for step, task in enumerate(gen_test):
        print(f'Batch {step}/{gen_test.num_tasks}')

        x, y = task['x_context'], task['y_context']
        x_test, y_test = task['x_target'], task['y_target']

        # For plotting MNIST ground truth image.
        image = task['image'] if args.data in ['mnist', 'fmnist', 'climate'] else None
        if args.data in ['mnist', 'fmnist']:
            target_channel = 0  # Only one channel.
        elif args.data == 'climate':
            target_channel = 1  # Temperature is second channel.
        else:
            target_channel = None

        if post_optimise:
            figdir = 'eval_figs_post_opt'
        else:
            figdir = 'eval_figs_no_post_opt'

        # Only plot the first batch.
        plot = True if step == 0 else False
        eval_dict = batch_eval_dataset(model, x, y, x_test, y_test,
                                       image=image,
                                       plot=plot,
                                       epoch=1,
                                       wd=wd,
                                       figdir=figdir,
                                       optimise=post_optimise,
                                       iters=args.post_iters,
                                       learning_rate=args.post_learning_rate,
                                       target_channel=target_channel,
                                       data_gen=gen_test,
                                       verbose=True)

        if step == 0:
            emp_risks = eval_dict['emp_risk']
            bounds = eval_dict['bounds']
            gen_risks = eval_dict['gen_risk']
            trivial_risks = eval_dict['trivial_risk']
            post_opt_successes = eval_dict['post_opt_success']
            optimistic_bounds = extract_metric('optimistic_bounds', eval_dict)
            binomial_tail_bounds = extract_metric('binomial_tail_bounds', eval_dict)
        else:
            emp_risks = np.concatenate([emp_risks, eval_dict['emp_risk']])
            bounds = np.concatenate([bounds, eval_dict['bounds']])
            gen_risks = np.concatenate([gen_risks, eval_dict['gen_risk']])
            trivial_risks = np.concatenate([trivial_risks, eval_dict['trivial_risk']])
            post_opt_successes = np.concatenate([post_opt_successes, eval_dict['post_opt_success']])
            optimistic_bounds = append_metric(optimistic_bounds, 'optimistic_bounds', eval_dict)
            binomial_tail_bounds = append_metric(binomial_tail_bounds, 'binomial_tail_bounds', eval_dict)

    return emp_risks, bounds, gen_risks, trivial_risks, post_opt_successes, optimistic_bounds, binomial_tail_bounds


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


def eval_and_save(args, wd, gen_test, model, post_optimise):
    emp_risks, bounds, gen_risks, trivial_risks, post_opt_successes, optimistic_bounds, binomial_tail_bounds = \
        eval(args, wd, gen_test, model, post_optimise=post_optimise)

    # Save metrics.
    names = ['emp_risk', 'bound', 'gen_risk', 'trivial_risk',
             'post_opt_success', 'optimistic_bounds', 'binomial_tail_bounds']
    eval_lists = [emp_risks, bounds, gen_risks, trivial_risks,
                  post_opt_successes, optimistic_bounds, binomial_tail_bounds]
    metrics = save_metrics(names, eval_lists)

    if post_optimise:
        title = 'eval_metrics_post_opt.txt'
    else:
        title = 'eval_metrics_no_post_opt.txt'

    eval_file = wd.file(title)
    with open(eval_file, 'w') as f:
        json.dump(metrics, f, indent=2)
