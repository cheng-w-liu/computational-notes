import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from kernels import RBFKernel
from gaussian_process import GaussianProcess


class BayesianOptimization(object):

    def __init__(self, score_func, bounds, policy='ei', epsilon=1e-7, lambda_val=1.5, gp_params=None):

        assert policy == 'ei' or policy =='ucb'

        self.score_func = score_func
        self.bounds = bounds
        self.policy = policy
        self.epsilon = epsilon
        self.lambda_val = lambda_val  # for ucb policy only
        if gp_params is not None:
            self.gp = GaussianProcess(**gp_params)
        else:
            n_params = bounds.shape[0]
            length_scale = 0.5 * np.ones(n_params)
            bounds = np.tile(np.array([1e-2, 1e2]), (n_params, 1))
            kernel = RBFKernel(length_scale=length_scale, length_scale_bounds=bounds)
            self.gp = GaussianProcess(kernel, alpha=0.03)

    def clone(self):
        cloned_obj = BayesianOptimization(self.score_func, self.bounds, self.policy,
                                          self.epsilon, self.lambda_val)
        cloned_obj.gp = self.gp.clone()
        return cloned_obj

    def fit(self, n_iter=10, x0=None, n_pre_samples=5, random_search=False):
        """
        Apply Bayesian Optimization to find the optimal parameter
        """
        if x0 is None:
            assert n_pre_samples is not None and n_pre_samples > 0

        if random_search:
            assert random_search > 1

        n_params = self.bounds.shape[0]

        x_list = []
        y_list = []

        if x0 is None:
            for params in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_pre_samples, n_params)):
                x_list.append(params)
                y_list.append(self.score_func(params))
        else:
            for params in x0:
                x_list.append(params)
                y_list.append(self.score_func(params))

        X = np.atleast_2d(np.array(x_list))
        y = np.array(y_list)

        for i in range(n_iter):

            self.gp.fit(X, y)

            if random_search:
                x_candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(random_search, n_params))
                acquisitions = -self.acquisition_function(x_candidates, y, n_params, self.policy)
                next_sample = x_candidates[np.argmax(acquisitions)]
            else:
                next_sample = self.sample_next_hyperparameter(self.acquisition_function, y, n_restart=10, policy=self.policy)

            if np.any(np.abs(next_sample - X) <= self.epsilon):
                next_sample = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

            x_list.append(next_sample)
            y_list.append(self.score_func(next_sample))

            X = np.atleast_2d(np.array(x_list))
            y = np.array(y_list)

        self.X_search = X
        self.y_search = y

    def optimal(self):
        return self.X_search[np.argmax(self.y_search)], np.max(self.y_search)

    def get_iteration_history(self):
        return self.X_search, self.y_search

    def acquisition_function(self, X, y, n_params, policy):

        if policy == 'ei':
            return self.negative_expected_improvement(X, y, n_params)
        elif self.policy == 'ucb':
            return self.negative_upper_confidence_bound(X, y, n_params)
        else:
            raise ValueError("unknown policy {0:}".format(self.policy))

    def negative_expected_improvement(self, X, y, n_params):

        X = np.reshape(X, (-1, n_params))

        mu, Sigma = self.gp.predict(X, return_cov=True)
        sigma = np.sqrt(np.diag(Sigma))

        mu = mu.ravel()
        sigma = sigma.ravel()

        f_best = np.max(y)
        Z = (mu - f_best) / sigma
        ei = (mu - f_best) * norm.cdf(Z) + sigma * norm.pdf(-Z)
        ei[sigma == 0.0] = 0.0
        return -ei

    def negative_upper_confidence_bound(self, X, y, n_params):

        X = np.reshape(X, (-1, n_params))

        mu, Sigma = self.gp.predict(X, return_cov=True)
        sigma = np.sqrt(np.diag(Sigma))

        mu = mu.ravel()
        sigma = sigma.ravel()

        ucb = mu + self.lambda_val * sigma
        return -ucb

    def sample_next_hyperparameter(self, acquisition_function, y, n_restart, policy):

        n_params = self.bounds.shape[0]
        best_x = None
        best_acquisition_value = 100.0

        for initial_value in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_restart, n_params)):
            res = minimize(fun=acquisition_function,
                           x0=initial_value,
                           bounds=self.bounds,
                           method='L-BFGS-B',
                           args=(y, n_params, policy))

            if res.fun < best_acquisition_value:
                best_acquisition_value = res.fun
                best_x = res.x

        return best_x


class BayesianOptimization1dDemo(object):

    def __init__(self, score_func, bounds, policy='ei', epsilon=1e-7, lambda_val=1.5, gp_params=None):

        assert policy == 'ei' or policy =='ucb'

        self.score_func = score_func
        self.bounds = bounds
        self.policy = policy
        self.epsilon = epsilon
        self.lambda_val = lambda_val  # for ucb policy only
        if gp_params is not None:
            self.gp = GaussianProcess(**gp_params)
        else:
            n_params = bounds.shape[0]
            length_scale = 0.5 * np.ones(n_params)
            bounds = np.tile(np.array([1e-2, 1e2]), (n_params, 1))
            kernel = RBFKernel(length_scale=length_scale, length_scale_bounds=bounds)
            self.gp = GaussianProcess(kernel, alpha=0.03)

    def fit(self, n_iter=10, x0=None, n_pre_samples=5, random_search=False):
        """
        Apply Bayesian Optimization to find the optimal parameter
        """
        if x0 is None:
            assert n_pre_samples is not None and n_pre_samples > 0

        if random_search:
            assert random_search > 1

        n_params = self.bounds.shape[0]

        x_list = []
        y_list = []

        if x0 is None:
            for params in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_pre_samples, n_params)):
                x_list.append(params)
                y_list.append(self.score_func(params))
        else:
            for params in x0:
                x_list.append(params)
                y_list.append(self.score_func(params))

        X = np.atleast_2d(np.array(x_list))
        y = np.array(y_list)

        evaluated_x_points = np.linspace(self.bounds[0][0], self.bounds[0][1], 200)
        evaluated_y_points = self.score_func(evaluated_x_points)

        for i in range(n_iter):

            self.gp.fit(X, y)

            if random_search:
                x_candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(random_search, n_params))
                acquisitions = -self.acquisition_function(x_candidates, y, n_params, self.policy)
                next_sample = x_candidates[np.argmax(acquisitions)]
            else:
                next_sample = self.sample_next_hyperparameter(self.acquisition_function, y, n_restart=10, policy=self.policy)

            # -------------------------------------------------------------------------------------------- #
            # plotting info
            if i > 0:

                # function values
                y_posterior, Sigma = self.gp.predict(np.reshape(evaluated_x_points, (-1, n_params)), return_cov=True)
                sigma = np.sqrt(np.diag(Sigma))

                acquisitions = self.acquisition_function(evaluated_x_points, y, n_params, self.policy)

                selected_point = [next_sample, self.gp.predict(np.reshape(next_sample, (-1, n_params)), return_cov=False)]

                # acquisition function values
                #acquisition = self.acquisition_function(next_sample, y, n_params, self.policy)
                #acquired_point = [next_sample, acquisition]
                acquired_point = [next_sample, self.acquisition_function(next_sample, y, n_params, self.policy)]

                # plot
                plot_progress(i, evaluated_x_points, evaluated_y_points, X, y, y_posterior, sigma, selected_point,
                              acquisitions, acquired_point, prefix=self.policy + "_")
            # -------------------------------------------------------------------------------------------- #

            if np.any(np.abs(next_sample - X) <= self.epsilon):
                next_sample = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

            x_list.append(next_sample)
            y_list.append(self.score_func(next_sample))

            X = np.atleast_2d(np.array(x_list))
            y = np.array(y_list)

            self.X_search = X
            self.y_search = y

    def optimal(self):
        return self.X_search[np.argmax(self.y_search)], np.max(self.y_search)

    def acquisition_function(self, X, y, n_params, policy):

        if policy == 'ei':
            return self.negative_expected_improvement(X, y, n_params)
        elif self.policy == 'ucb':
            return self.negative_upper_confidence_bound(X, y, n_params)
        else:
            raise ValueError("unknown policy {0:}".format(self.policy))

    def negative_expected_improvement(self, X, y, n_params):

        X = np.reshape(X, (-1, n_params))

        mu, Sigma = self.gp.predict(X, return_cov=True)
        sigma = np.sqrt(np.diag(Sigma))

        mu = mu.ravel()
        sigma = sigma.ravel()

        f_best = np.max(y)
        Z = (mu - f_best) / sigma
        ei = (mu - f_best) * norm.cdf(Z) + sigma * norm.pdf(-Z)
        ei[sigma == 0.0] = 0.0
        return -ei

    def negative_upper_confidence_bound(self, X, y, n_params):

        X = np.reshape(X, (-1, n_params))

        mu, Sigma = self.gp.predict(X, return_cov=True)
        sigma = np.sqrt(np.diag(Sigma))

        mu = mu.ravel()
        sigma = sigma.ravel()

        ucb = mu + self.lambda_val * sigma
        return -ucb

    def sample_next_hyperparameter(self, acquisition_function, y, n_restart, policy):

        n_params = self.bounds.shape[0]
        best_x = None
        best_acquisition_value = 100.0

        for initial_value in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_restart, n_params)):
            res = minimize(fun=acquisition_function,
                           x0=initial_value,
                           bounds=self.bounds,
                           method='L-BFGS-B',
                           args=(y, n_params, policy))

            if res.fun < best_acquisition_value:
                best_acquisition_value = res.fun
                best_x = res.x

        return best_x


def plot_progress(iter_idx, evaluated_x_points, evaluated_y_points, X, y, y_posterior, sigma, selected_point,
                  acquisitions, acquired_point, prefix=''):

    import matplotlib
    import matplotlib.pyplot as plt
    from plot_functions import adjustAxeProperties

    matplotlib.style.use('ggplot')

    evaluated_x_points = evaluated_x_points.ravel()
    y_posterior = y_posterior.ravel()

    FONTSIZE = 23
    plt.close('all')
    fig = plt.figure(figsize=(12, 16))  # horizontal, vertical
    gs = matplotlib.gridspec.GridSpec(2, 1)  # vertical, horizontal

    # plot function, GP posterior, and the optimal point
    ax = plt.subplot(gs[0, 0])
    ax.plot(evaluated_x_points, evaluated_y_points, label='true function', linestyle='-', linewidth=2, color='k')
    ax.plot(evaluated_x_points, y_posterior, label='posterior mean', linestyle='-.', linewidth=1.5, color='b')
    ax.fill_between(evaluated_x_points, y_posterior-2*sigma, y_posterior+2*sigma, label='GP posterior', alpha=0.5, color='0.75')
    ax.scatter(X.ravel(), y, label='observed', marker='X', s=100, color='g')
    ax.scatter(selected_point[0], selected_point[1], label='selected', marker="^", s=180, color='r')
    adjustAxeProperties(ax, FONTSIZE, 0, FONTSIZE, 0)
    ax.legend(loc='best', fontsize=FONTSIZE)
    ax.set_xlabel('x', fontsize=FONTSIZE, labelpad=15)
    ax.set_ylabel('f(x)', fontsize=FONTSIZE, labelpad=15)

    # plot the acquisition values
    ax = plt.subplot(gs[1, 0])
    ax.plot(evaluated_x_points, acquisitions, linestyle='-', linewidth=1.5, color='k')
    ax.scatter(acquired_point[0], acquired_point[1], label='Selected', marker="^", s=180, color='r')
    adjustAxeProperties(ax, FONTSIZE, 0, FONTSIZE, 0)
    ax.set_xlabel('x', fontsize=FONTSIZE, labelpad=15)
    ax.set_ylabel('Acquisitions', fontsize=FONTSIZE, labelpad=15)

    plt.tight_layout(pad=0, w_pad=1.0, h_pad=2.0)
    fig.suptitle("Iteration: {0:}".format(iter_idx), fontsize=1.05*FONTSIZE)
    plt.subplots_adjust(top=0.93)
    fname = str(iter_idx).zfill(2)
    plt.savefig(prefix + "iteration_{0:}.png".format(fname), bbox_inches='tight')
