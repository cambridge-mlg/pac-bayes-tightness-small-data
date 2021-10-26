import math
import abc
import torch
import torch.nn as nn
from torch.optim import Adam
from .utils import device, GaussianIntegrator, _e_bar
from .convex import BernoulliKL


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def sigmoid_squared(x):
    return sigmoid(x) ** 2


def post_params_clone(params_dict):
    return {k: torch.clone(t).detach() for k, t in params_dict.items()}


class Classifier(nn.Module):
    """Base class for meta learning models that use linear Gaussian
    predictions with an MLP feature map and Deep Sets inference net.
    """
    def __init__(self, architecture=None, deterministic=False, delta=.1):
        super().__init__()
        self.architecture = architecture
        self.delta = torch.Tensor([delta]).to(device)
        self.loss_fn = 'classification'
        self.integrator = GaussianIntegrator(num_nodes=64)
        self.deterministic = deterministic

    def risk_from_predictive(self, pred_means, pred_vars, y, deterministic=False):
        """
        Args:
            pred_means: [B, N] torch, mean of predictive function at x
            pred_vars: [B, N] torch, var of predictive function at x
            y: [B, N, output_dim] torch, output datapoints to compute risk over
            deterministic: bool, If true, does deterministic majority vote
                classification.

        Returns:
            risk: [B] torch, risk on y averaged over the posterior
        """
        pred_vars = torch.clamp(pred_vars, min=1e-6)
        if self.loss_fn == 'classification':
            # y is in {-1, +1}
            if deterministic:  # Harden the classifier.
                risk = (1. - torch.sign(y[:, :, 0] * torch.sign(pred_means))) / 2  # [B, N]
            else:
                numerator = -y[:, :, 0] * pred_means  # [B, N]
                denominator = torch.sqrt(2. * pred_vars)  # [B, N]
                risk = 0.5 * (1. + torch.erf(numerator / denominator))  # [B, N]
        elif self.loss_fn == 'regression':
            # y is in range [0, +1]
            targets = y[:, :, 0]  # [B, N]
            pred_std = torch.sqrt(pred_vars)

            def squared_error(x):
                return (sigmoid(x) - targets[..., None]) ** 2
            mean_squared_error = self.integrator.integrate(squared_error,
                                                      pred_means, pred_std)  # [B, N]
            risk = mean_squared_error  # [B, N]
        elif self.loss_fn == 'nll':
            targets = y[:, :, 0]  # [B, N]
            nll = (targets - pred_means) ** 2 / pred_vars + torch.log(pred_vars)  # [B, N]
            risk = nll  # NLL up to an additive and multiplicative constant.
        else:
            raise NotImplementedError
        assert torch.all(torch.isfinite(risk))
        risk = risk.mean(dim=-1)  # [B]
        return risk

    def risks(self, xy_dict, post, deterministic=False):
        """From a dictionary of xy values, return a dictionary of risks."""
        # xy[0] is x values, xy[1] is y values
        feature_map = self.architecture.feature_map
        preds = {k: self.architecture.predictive(post, feature_map, xy[0])
                 for k, xy in xy_dict.items()}
        # preds[k] is pred_means, pred_vars for key k.
        risks = {k: self.risk_from_predictive(*preds[k], xy[1],
                                              deterministic=deterministic)
                 for k, xy in xy_dict.items()}
        return risks

    def class_probs(self, pred_means, pred_vars, deterministic=False):
        """
        Args:
            pred_means: [B, N_grid] torch, mean of predictive function at x
            pred_vars: [B, N_grid] torch, var of predictive function at x
            deterministic: bool, If true, does deterministic majority vote
                classification.

        Returns:
            class_probs: [B, N_grid] torch, probability that classifier sampled
                from the posterior would give a prediction of +1 at x
        """
        if deterministic:
            class_probs = (torch.sign(pred_means) + 1.) / 2.  # [B, N]
        else:
            numerator = pred_means  # [B, N]
            denominator = torch.sqrt(2. * pred_vars)  # [B, N]
            class_probs = 0.5 * (1 + torch.erf(numerator / denominator))  # [B, N]
        return class_probs

    def reg_probs(self, pred_means, pred_vars):
        """
        Args:
            pred_means: [B, N_grid] torch, mean of predictive function at x
            pred_vars: [B, N_grid] torch, var of predictive function at x

        Returns:
            reg_means: [B, N_grid] torch, mean of predictive distribution after
                being squashed through a sigmoid.
            reg_vars: [B, N_grid] torch, variance of predictive distribution
                after being squashed through a sigmoid.
        """
        pred_std = torch.sqrt(pred_vars)
        mean_sigmoid = self.integrator.integrate(sigmoid, pred_means, pred_std)  # [B, N]

        def squared_deviation(x):
            return (sigmoid(x) - mean_sigmoid[..., None]) ** 2
        var_sigmoid = self.integrator.integrate(squared_deviation,
                                                pred_means, pred_std)  # [B, N]
        return mean_sigmoid, var_sigmoid

    def params_grad(self, has_grad):
        """ Freeze or unfreeze gradients for all the learnable parameters in
        the model.
        Args:
            has_grad: bool, freezes if true.
        """
        for param in self.parameters():
            param.requires_grad = has_grad

    def batch_post_optimise(self, x, y, iters=3000, learning_rate=1e-3, verbose=True):
        """Optimise a *batch* of posteriors at test time using gradient based methods,
        after meta-learning is complete.
        Args:
            x: [B, N, input_dim] torch, input datapoints
            y: [B, N, output_dim] torch, output datapoints
            iters: int, number of gradient steps
            learning_rate: float
        """
        # Freeze all parameters we do not want to optimise
        self.params_grad(has_grad=False)

        # Split the data for post optimisation.
        x_infer_init, y_infer_init, x_compute_loss, y_compute_loss, no_post_opt = \
            self.post_opt_data_split(x, y)

        # Get the posterior as initialisation, [B, feature_dim]
        post = self.architecture.infer(x_infer_init, y_infer_init)

        if no_post_opt:  # Validation model with an empty train set should not be optimised.
            post_opt_success = False
            print("Train set is empty in a validation model. Not performing "
                  "post optimisation.")
            return post, post_opt_success

        # Save parameters of the entire batch in case post optimisation fails.
        stored_params = post_params_clone(post.params)
        post.set_learnable()

        opt = Adam(post.learnable_param_list(), lr=learning_rate)

        lowest_bound = math.inf
        post_opt_success = True
        for i in range(iters):
            try:
                loss_terms = self.post_loss(post, x_compute_loss, y_compute_loss)  # []
            except AssertionError:
                print('Post-optimisation has encountered numerical issues. '
                      'Reverting to non-post-optimised parameters.')
                post.fix_learned_params()
                self.params_grad(has_grad=True)
                post.params = stored_params
                post_opt_success = False
                break
            loss = loss_terms['post_optimisation_loss']
            loss.backward()
            opt.step()
            opt.zero_grad()
            if i % 100 == 0:
                bound = loss_terms['risk_bounds']
                if verbose:
                    print(f'Average bound over batch of {bound.shape[0]}'
                          f' after {i} iters: {bound.mean().item()}')
                if bound.mean().item() >= lowest_bound - 0.0001:
                    break  # break early if no progress is being made
                lowest_bound = min(lowest_bound, bound.mean().item())

        # Freeze the learned posterior parameters
        post.fix_learned_params()

        # Unfreeze network parameters in case further learning is to be done
        self.params_grad(has_grad=True)

        return post, post_opt_success


class ValClassifier(Classifier):
    def __init__(self, val_proportion=.2, **kwargs):
        super().__init__(**kwargs)
        self.val_proportion = val_proportion
        self.N_val = None

    def binomial_tail_bound(self, val_risk):
        """
        Compute the binomial tail risk bound from Langford.
        Args:
            val_risk: [B], torch

        Returns:
            bin tail bound: [B], torch
        """
        return _e_bar(self.delta, val_risk, self.N_val)

    def loss(self, x, y):
        """Returns the average risk on the validation set over the batch.
        Args:
            x: [B, N, input_dim] torch, input datapoints
            y: [B, N, output_dim] torch, output datapoints

        Returns:
            loss: [] torch, average validation risk
            risk_bounds: [B] torch, risk bounds
            emp_risk: [B] torch, train risks
            val_risk: [B] torch, val risks
        """

        # Split the data, [B, N_train / N_val, input_dim / output_dim]
        x_train, y_train, x_val, y_val = self.data_split(x, y)

        # Get posterior over weights, [B, feature_dim]
        post = self.architecture.infer(x_train, y_train)

        return self.post_loss(post, x, y)

    def post_loss(self, post, x, y):
        """
        Compute the empirical risk for post-hoc optimisation of the validation
        model. N.B. this may lead to overfitting.
        Args:
            post: dist object, batch of B
            x: [B, N, input_dim] torch, input datapoints
            y: [B, N, output_dim] torch, output datapoints

        Returns:
            loss: [] torch, log of average Catoni risk bound over batch
        """

        # Split the data, [B, N_train / N_val, input_dim / output_dim]
        x_train, y_train, x_val, y_val = self.data_split(x, y)
        self.N_val = x_val.shape[1]

        # Get validation predictive, [B, N_val]
        feature_map = self.architecture.feature_map
        pred_means, pred_vars = self.architecture.predictive(post,
                                                         feature_map, x_val)

        # Get train predictive, [B, N_train]
        pred_means_train, pred_vars_train = self.architecture.predictive(post,
                                                         feature_map, x_train)

        # Compute training risk for logging, [B]
        train_risk = self.risk_from_predictive(pred_means_train,
                                               pred_vars_train, y_train)

        # Compute the validation risk, [B]
        val_risk = self.risk_from_predictive(pred_means, pred_vars, y_val)

        # Compute average batch loss for backpropagation
        loss = torch.mean(val_risk)  # []

        # Compute risk bounds
        risk_bounds = self.risk_bounds(val_risk)  # [B]

        # Take mean for post optimisation loss so gradients in batch don't
        # interfere with each other.
        return {'loss': loss, 'risk_bounds': risk_bounds,
                'emp_risk': train_risk, 'val_risk': val_risk,
                'post_optimisation_loss': torch.mean(train_risk)}

    def post_opt_data_split(self, x, y):
        """
        Perform data split for post optimisation.
        Args:
            x: Entire context set inputs
            y: Entire context set outputs
        Returns:
            x_infer_init, y_infer_init: Dataset used by inference network to
                initialise the posterior.
            x_compute_loss, y_compute_loss: Dataset used by post_loss to compute
                the loss of the posterior. post_loss for the validation model
                requires the entire context set as an input.
            no_post_opt, bool: If True, do not post optimise. This is the case
                when train set is zero since then empirical risk is NaN
                (undefined).
        """
        x_train, y_train, x_val, y_val = self.data_split(x, y)
        x_infer_init = x_train
        y_infer_init = y_train
        x_compute_loss = x
        y_compute_loss = y

        if x_train.numel() == 0:  # Empty train set for validation model.
            no_post_opt = True
        else:
            no_post_opt = False

        return x_infer_init, y_infer_init, x_compute_loss, y_compute_loss, no_post_opt

    def evaluate(self, x_context, y_context, x_target, y_target, x_grid,
                 post_optimise=False, **kwargs):
        """For evaluation and plotting on a task, using a validation set.
        Args:
            x_context: [B, N, input_dim] torch, input datapoints
            y_context: [B, N, output_dim] torch, output datapoints
            x_target: [B, N_target, input_dim] torch, query locations
            y_target: [B, N_target, output_dim] torch, query class labels
            x_grid: [B, N_grid, input_dim] torch, grid of input locations for
                plotting purposes

        Returns:
            N.B. All of the following are computed for the deterministic version
            of the classifier.

            train_risk: [B] torch, train risks
            gen_risk: [B] torch, generalisation risks, estimated based on the
                target set
            bounds: [B] torch, validation bounds on generalisation risk
            class_probs: [B, N_grid] torch, probability that classifier sampled
                from the posterior returns +1 at locations in x_grid
        """
        batch_size = x_context.shape[0]

        # Split the data, [B, N_train / N_val, input_dim / output_dim]
        x_train, y_train, x_val, y_val = self.data_split(x_context, y_context)
        self.N_val = x_val.shape[1]

        xy_dict = {'train': [x_train, y_train],
                   'val': [x_val, y_val],
                   'gen': [x_target, y_target]}

        # Get posterior over weights, [B, feature_dim]
        if post_optimise:
            post, post_opt_success = self.batch_post_optimise(
                                                x_context, y_context, **kwargs)
        else:
            post_opt_success = False
            post = self.architecture.infer(x_train, y_train)

        # # Get posterior over weights, [B, feature_dim]
        # post = self.architecture.infer(x_train, y_train)

        # Compute risks, dict of [B]
        risk_dict = self.risks(xy_dict, post, deterministic=True)

        # Compute class probabilities at x_grid for plotting, [B, N_grid]
        feature_map = self.architecture.feature_map
        grid_pred_means, grid_pred_vars = self.architecture.predictive(post,
                                                           feature_map, x_grid)
        class_probs = self.class_probs(grid_pred_means, grid_pred_vars,
                                       deterministic=True)

        # Compute sigmoid regression moments for plotting, [B, N_grid]
        # N.B. there is no deterministic version for regression.
        sigmoid_means, sigmoid_vars = self.reg_probs(grid_pred_means,
                                                     grid_pred_vars)

        # Compute validation bound based on validation set, [B]
        bounds = self.risk_bounds(risk_dict['val'])

        # Post optimisation successes: either all fail or all succeed.
        post_opt_success = torch.tensor([post_opt_success] * batch_size)

        # Compute binomial tail bounds, [B]
        bin_tail_bound = self.binomial_tail_bound(risk_dict['val'])

        out = {'emp_risk': risk_dict['train'],
               'val_risk': risk_dict['val'],
               'gen_risk': risk_dict['gen'],
               'bounds': bounds,
               'class_probs': class_probs,
               'sigmoid_means': sigmoid_means,
               'sigmoid_vars': sigmoid_vars,
               'post_opt_success': post_opt_success,
               'binomial_tail_bounds': bin_tail_bound}
        return out

    def data_split(self, x, y):
        """Splits dataset into N_train and N_val sized subsets, where
        N = N_train + N_val. N.B. this is done in such as a way as to ensure
        that the train set for validation coincides with the prior set for the
        DDP PAC-Bayes models.
        Args:
            x: [B, N, input_dim] torch, input datapoints
            y: [B, N, output_dim] torch, output datapoints

        Returns:
            x_train: [B, N_train, input_dim] torch, input datapoints that
                posterior can depend on
            y_train: [B, N_train, input_dim] torch, output datapoints that
                posterior can depend on
            x_val: [B, N_val, input_dim] torch, input datapoints for the
                validation bound
            y_val: [B, N_val, input_dim] torch, output datapoints for the
                validation bound
        """
        N = x.shape[1]
        train_proportion = 1. - self.val_proportion
        N_train = round(N * train_proportion)

        x_train, y_train = x[:, :N_train, :], y[:, :N_train, :]
        x_val, y_val = x[:, N_train:, :], y[:, N_train:, :]

        return x_train, y_train, x_val, y_val


class PACBayesClassifier(Classifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Prior is a trainable network parameter
        self.prior = self.architecture.dist_family.instantiate_dist(
            batch_size=1,
            feature_dim=self.architecture.feature_dim,
            is_network_parameter=True
        )
        self.kl_delta = BernoulliKL()  # For computing optimistic bounds.

    @abc.abstractmethod
    def risk_bounds(self, emp_risk, KL, N):
        """
        Compute the PAC-Bayes risk bound.
        Args:
            emp_risk: [B], torch
            KL: [B], torch
            N: int

        Returns:
            bounds: [B], torch
        """
    
    @abc.abstractmethod
    def batch_losses(self, emp_risk, KL, N):
        """
        Compute loss function for every dataset in the batch.
        Args:
            emp_risk: [B], torch
            KL: [B], torch
            N: int

        Returns:
            log_bounds: [B], torch
        """

    def optimistic_risk_bounds(self, emp_risk, KL, N):
        """
        Compute the optimistic risk bound "illegal Maurer bound", obtained by
        removing the log(2 sqrt(N)) term from the Maurer bound.
        Args:
            emp_risk: [B], torch
            KL: [B], torch
            N: int

        Returns:
            optimistic bounds: [B], torch
        """
        """Compute the Maurer risk bound."""
        rhs = (KL + torch.log(1 / self.delta)) / N
        return self.kl_delta.biggest_inverse(emp_risk, rhs)  # [B]


    def loss(self, x, y):
        """
        Perform forward pass through inference network to get the posterior,
        then compute losses and metrics.
        """
        post = self.architecture.infer(x, y)
        return self.post_loss(post, x, y)

    def post_loss(self, post, x, y, compute_optimistic=False):
        """
        Compute log of average of the PAC Bayes generalisation risk bounds over
        the batch as the loss, along with the actual bound values.
        Args:
            post: dist object, batch of B
            x: [B, N, input_dim] torch, input datapoints
            y: [B, N, output_dim] torch, output datapoints
            compute_optimistic: bool, if True, computes optimistic bounds as
                well.

        Returns:
            loss: [] torch, log of average Catoni risk bound over batch
            risk_bounds: [B] torch, Catoni risk bounds
            emp_risk: [B] torch, empirical risks
            kl: [B] torch, KL terms
        """
        batch_size = x.shape[0]
        N = x.shape[1]

        # Get predictive over functions, [B, N]
        feature_map = self.architecture.feature_map
        pred_means, pred_vars = self.architecture.predictive(post, feature_map, x)

        # Compute the empirical risk, [B]
        emp_risk = self.risk_from_predictive(pred_means, pred_vars, y)

        # Compute KL term, [B] - N.B. prior var is fixed to 1
        KL = self.architecture.kl(post, self.prior)

        # Compute losses, [B]
        batch_losses = self.batch_losses(emp_risk, KL, N)

        # Aggregate losses, [] N.B. batch_losses assumed to be in log space.
        loss = torch.logsumexp(batch_losses, dim=0) - math.log(batch_size)

        # Compute risk bounds, [B]
        risk_bounds = self.risk_bounds(emp_risk, KL, N)

        # Compute loss for post optimisation, []
        # Take mean so that gradients for each element of the batch do not
        # interfere with each other during batched post optimisation.
        post_optimisation_loss = torch.mean(batch_losses)

        loss_dict = {'loss': loss,
                     'risk_bounds': risk_bounds,
                     'emp_risk': emp_risk,
                     'kl': KL,
                     'post_optimisation_loss': post_optimisation_loss}

        # Compute optimistic risk bounds, [B]
        if compute_optimistic:
            optimistic_bounds = self.optimistic_risk_bounds(emp_risk, KL, N)
            loss_dict['optimistic_bounds'] = optimistic_bounds

        return loss_dict

    def post_opt_data_split(self, x, y):
        """
        Perform data split for post optimisation.
        Args:
            x: Entire context set inputs
            y: Entire context set outputs
        Returns:
            x_infer_init, y_infer_init: Dataset used by inference network to
                initialise the posterior. PAC-Bayes models can use entire
                dataset, as opposed to validation models.
            x_compute_loss, y_compute_loss: Dataset used by post_loss to compute
                the loss of the posterior.
            no_post_opt, bool: If true, do not post optimise. This is never true
                for PAC-Bayes.
        """
        x_infer_init = x
        y_infer_init = y
        x_compute_loss = x
        y_compute_loss = y

        no_post_opt = False  # PAC Bayes model can always be post optimised, even if prior set is empty

        return x_infer_init, y_infer_init, x_compute_loss, y_compute_loss, no_post_opt

    def evaluate(self, x_context, y_context, x_target, y_target, x_grid,
                 post_optimise=False, **kwargs):
        """For evaluation and plotting on a task.
        Args:
            x_context: [B, N, input_dim] torch, input datapoints
            y_context: [B, N, output_dim] torch, output datapoints
            x_target: [B, N_target, input_dim] torch, query locations
            y_target: [B, N_target, output_dim] torch, query class labels
            x_grid: [B, N_grid, input_dim] torch, grid of input locations for
                plotting purposes
            post_optimise: bool, if True, optimise the posterior with gradient
                based methods after meta-training. Requires batch size B = 1.
            **kwargs: iters, learning_rate: arguments for
                post-meta-training optimiser

        Returns:
            emp_risk: [B] torch, empirical risks
            gen_risk: [B] torch, generalisation risks, estimated based on the
                target set
            bounds: [B] torch, Catoni bounds on generalisation risk
            class_probs: [B, N_grid] torch, probability that classifier sampled
                from the posterior returns +1 at locations in x_grid
            KL: [B] torch, KL divergence terms
        """
        batch_size = x_context.shape[0]
        N = x_context.shape[1]

        # Get posterior over weights, [B, feature_dim]
        if post_optimise:
            post, post_opt_success = self.batch_post_optimise(
                                            x_context, y_context, **kwargs)
        else:
            post_opt_success = False
            post = self.architecture.infer(x_context, y_context)

        # Compute generalisation risks, dict of [B]
        xy_dict = {'gen': [x_target, y_target]}
        risk_dict = self.risks(xy_dict, post)

        # Compute losses based on context set
        loss_dict = self.post_loss(post, x_context, y_context,
                                   compute_optimistic=True)

        # Compute class probabilities at x_grid for plotting, [B, N_grid]
        feature_map = self.architecture.feature_map
        grid_pred_means, grid_pred_vars = self.architecture.predictive(post,
                                                           feature_map, x_grid)
        class_probs = self.class_probs(grid_pred_means, grid_pred_vars)

        # Compute sigmoid regression moments for plotting, [B, N_grid]
        sigmoid_means, sigmoid_vars = self.reg_probs(grid_pred_means,
                                                     grid_pred_vars)

        # Post optimisation successes: either all fail or all succeed.
        post_opt_success = torch.tensor([post_opt_success] * batch_size)

        out = {'emp_risk': loss_dict['emp_risk'],
               'gen_risk': risk_dict['gen'],
               'bounds': loss_dict['risk_bounds'],
               'optimistic_bounds': loss_dict['optimistic_bounds'],
               'class_probs': class_probs,
               'KL': loss_dict['kl'],
               'sigmoid_means': sigmoid_means,
               'sigmoid_vars': sigmoid_vars,
               'post_opt_success': post_opt_success}
        return out


class PACBayesDDPClassifier(PACBayesClassifier):
    """Use data dependent priors, where the prior depends on a proportion
    prior_proportion of the dataset"""
    def __init__(self, prior_proportion, **kwargs):
        super().__init__(**kwargs)
        self.prior_proportion = prior_proportion
        # Don't train prior for DDP since the prior mean is output by an
        # inference network.
        self.prior = None

    def data_split(self, x, y):
        """Splits dataset into N_prior and N_risk sized subsets, where
        N = N_prior + N_risk.
        Args:
            x: [B, N, input_dim] torch, input datapoints
            y: [B, N, output_dim] torch, output datapoints

        Returns:
            x_prior: [B, N_prior, input_dim] torch, input datapoints that prior
                can depend on
            y_prior: [B, N_prior, input_dim] torch, output datapoints that prior
                can depend on
            x_risk: [B, N_risk, input_dim] torch, input datapoints for the
                empirical risk term
            y_risk: [B, N_risk, input_dim] torch, output datapoints for the
                empirical risk term
        """
        N = x.shape[1]
        N_prior = round(N * self.prior_proportion)

        x_prior, y_prior = x[:, :N_prior, :], y[:, :N_prior, :]
        x_risk, y_risk = x[:, N_prior:, :], y[:, N_prior:, :]

        return x_prior, y_prior, x_risk, y_risk

    def post_loss(self, post, x, y, compute_optimistic=False):
        """
        Compute log of average of the PAC Bayes generalisation risk bounds over
        the batch as the loss, along with the actual bound values, using a
        data-dependent prior.
        Args:
            post: dist object, batch of B
            x: [B, N, input_dim] torch, input datapoints
            y: [B, N, output_dim] torch, output datapoints
            compute_optimistic: bool, if True, computes optimistic bounds as
                well.

        Returns:
            loss: [] torch, log of average Catoni risk bound over batch
            risk_bounds: [B] torch, Catoni risk bounds
            emp_risk: [B] torch, empirical risks
            kl: [B] torch, KL terms
            prior: dist object, batch of B
        """
        batch_size = x.shape[0]

        # Split the data for use by the prior and estimating empirical risk,
        # [B, N_prior *OR* N_risk, input_dim]
        x_prior, y_prior, x_risk, y_risk = self.data_split(x, y)

        # Compute the sample dependent prior, [B, feature_dim]
        prior = self.architecture.prior(x_prior, y_prior)

        # Get predictive over functions only at the empirical risk estimation
        # data points, [B, N_risk]
        feature_map = self.architecture.feature_map
        pred_means, pred_vars = self.architecture.predictive(post, feature_map,
                                                             x_risk)

        # Compute the empirical risk on the risk points, [B]
        emp_risk = self.risk_from_predictive(pred_means, pred_vars, y_risk)

        # Compute KLs between posterior and sample dependent prior, [B]
        KL = self.architecture.kl(post, prior)

        # Compute losses, [B]
        N_risk = x_risk.shape[1]
        batch_losses = self.batch_losses(emp_risk, KL, N_risk)

        # Aggregate losses, [] N.B. batch_losses assumed to be in log space.
        loss = torch.logsumexp(batch_losses, dim=0) - math.log(batch_size)  # []

        # Compute risk bounds, [B]
        risk_bounds = self.risk_bounds(emp_risk, KL, N_risk)

        # Compute loss for post optimisation, []
        # Take mean so that gradients for each element of the batch do not
        # interfere with each other during batched post optimisation.
        post_optimisation_loss = torch.mean(batch_losses)

        loss_dict = {'loss': loss,
                     'risk_bounds': risk_bounds,
                     'emp_risk': emp_risk,
                     'kl': KL,
                     'prior': prior,
                     'post_optimisation_loss': post_optimisation_loss}

        # Compute optimistic risk bounds, [B]
        if compute_optimistic:
            optimistic_bounds = self.optimistic_risk_bounds(emp_risk, KL, N_risk)
            loss_dict['optimistic_bounds'] = optimistic_bounds

        return loss_dict

    def evaluate(self, x_context, y_context, x_target, y_target, x_grid,
                 post_optimise=False, **kwargs):
        """For evaluation and plotting on a task with a data-dependent prior.
        Args:
            x_context: [B, N, input_dim] torch, input datapoints
            y_context: [B, N, output_dim] torch, output datapoints
            x_target: [B, N_target, input_dim] torch, query locations
            y_target: [B, N_target, output_dim] torch, query class labels
            x_grid: [B, N_grid, input_dim] torch, grid of input locations for
                plotting purposes
            post_optimise: bool, if True, optimise the posterior with gradient
                based methods after meta-training. Requires batch size B = 1.
            **kwargs: iters, learning_rate: arguments for
                post-meta-training optimiser

        Returns:
            emp_risk: [B] torch, empirical risks
            gen_risk: [B] torch, generalisation risks, estimated based on the
                target set
            bounds: [B] torch, Catoni bounds on generalisation risk
            class_probs: [B, N_grid] torch, probability that classifier sampled
                from the posterior returns +1 at locations in x_grid
            prior_class_probs: [B, N_grid] torch, probability that classifier
                sampled from the prior returns +1 at locations in x_grid
            KL: [B] torch, KL divergence terms
        """
        batch_size = x_context.shape[0]

        # Get posterior over weights, [B, feature_dim]
        if post_optimise:
            post, post_opt_success = self.batch_post_optimise(
                                            x_context, y_context, **kwargs)
        else:
            post_opt_success = False
            post = self.architecture.infer(x_context, y_context)

        # Compute generalisation risks, [B]
        xy_dict = {'gen': [x_target, y_target]}
        risk_dict = self.risks(xy_dict, post)

        # Compute losses based on context set
        loss_dict = self.post_loss(post, x_context, y_context,
                                   compute_optimistic=True)
        prior = loss_dict['prior']

        # Compute class probabilities at x_grid for plotting, [B, N_grid]
        feature_map = self.architecture.feature_map
        grid_pred_means, grid_pred_vars = self.architecture.predictive(post,
                                                           feature_map, x_grid)
        grid_prior_means, grid_prior_vars = self.architecture.predictive(
                                                    prior, feature_map, x_grid)

        prior_class_probs = self.class_probs(grid_prior_means, grid_prior_vars)
        class_probs = self.class_probs(grid_pred_means, grid_pred_vars)

        # Compute sigmoid regression moments for plotting, [B, N_grid]
        prior_sigmoid_means, prior_sigmoid_vars = self.reg_probs(
                                            grid_prior_means, grid_prior_vars)
        sigmoid_means, sigmoid_vars = self.reg_probs(
                                            grid_pred_means, grid_pred_vars)

        # Post optimisation successes: either all fail or all succeed.
        post_opt_success = torch.tensor([post_opt_success] * batch_size)

        out = {'emp_risk': loss_dict['emp_risk'],
               'gen_risk': risk_dict['gen'],
               'bounds': loss_dict['risk_bounds'],
               'optimistic_bounds': loss_dict['optimistic_bounds'],
               'class_probs': class_probs,
               'prior_class_probs': prior_class_probs,
               'KL': loss_dict['kl'],
               'sigmoid_means': sigmoid_means,
               'sigmoid_vars': sigmoid_vars,
               'prior_sigmoid_means': prior_sigmoid_means,
               'prior_sigmoid_vars': prior_sigmoid_vars,
               'post_opt_success': post_opt_success}
        return out
