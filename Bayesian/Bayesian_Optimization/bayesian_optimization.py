import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from kernels import RBFKernel
from gaussian_process import GaussianProcess

class BayesianOptimization(object):

    def __init__(self, score_func, bounds, epsilon=1e-7, gp_params=None):

        self.score_func = score_func
        self.bounds = bounds
        self.epsilon = epsilon
        if gp_params is not None:
            self.gp = GaussianProcess(**gp_params)
        else:
            n_params = bounds.shape[0]
            length_scale = 0.5 * np.ones(n_params)
            bounds = np.tile(np.array([1e-2, 1e2]), (n_params, 1))
            kernel = RBFKernel(length_scale=length_scale, length_scale_bounds=bounds)
            self.gp = GaussianProcess(kernel, alpha=0.02)

    def fit(self, n_iter=10, x0=None, n_pre_samples=10, random_search=False):
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

        for _ in range(n_iter):

            self.gp.fit(X, y)

            if random_search:
                x_candidates = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(random_search, n_params))
                ei = -self.negative_expected_improvement(x_candidates, y, n_params)
                next_sample = x_candidates[np.argmax(ei)]
            else:
                next_sample = self.sample_next_hyperparameter(self.negative_expected_improvement, y, n_restart=10)

            if np.any(np.abs(next_sample - X) <= self.epsilon):
                next_sample = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

            x_list.append(next_sample)
            y_list.append(self.score_func(next_sample))

            X = np.atleast_2d(np.array(x_list))
            y = np.array(y_list)

        return X, y

    def negative_expected_improvement(self, X, y, n_params):

        X = np.reshape(X, (-1, n_params))

        mu, Sigma = self.gp.predict(X, return_cov=True)
        sigma = np.sqrt(np.diag(Sigma))

        f_best = np.max(y)
        Z = (mu - f_best) / sigma
        ei = (mu - f_best) * norm.cdf(Z) + sigma * norm.pdf(-Z)
        ei[sigma == 0.0] = 0.0
        return -ei

    def sample_next_hyperparameter(self, acquisition_function, y, n_restart):

        n_params = self.bounds.shape[0]
        best_x = None
        best_acquisition_value = 100.0

        for initial_value in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_restart, n_params)):
            res = minimize(fun=acquisition_function,
                           x0=initial_value,
                           bounds=self.bounds,
                           method='L-BFGS-B',
                           args=(y, n_params))

            if res.fun < best_acquisition_value:
                best_acquisition_value = res.fun
                best_x = res.x

        return best_x


class BayesianOptimization1dDemo(object):

    def __init__(self, score_func, bounds, epsilon=1e-7, gp_params=None):

        self.score_func = score_func
        self.bounds = bounds
        self.epsilon = epsilon
        if gp_params is not None:
            self.gp = GaussianProcess(**gp_params)
        else:
            n_params = bounds.shape[0]
            length_scale = 0.5 * np.ones(n_params)
            bounds = np.tile(np.array([1e-2, 1e2]), (n_params, 1))
            kernel = RBFKernel(length_scale=length_scale, length_scale_bounds=bounds)
            self.gp = GaussianProcess(kernel, alpha=0.02)

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
                ei = -self.negative_expected_improvement(x_candidates, y, n_params)
                next_sample = x_candidates[np.argmax(ei)]
            else:
                next_sample = self.sample_next_hyperparameter(self.negative_expected_improvement, y, n_restart=10)

            # -------------------------------------------------------------------------------------------- #
            # plotting info
            if i > 0:
                y_posterior, Sigma = self.gp.predict(np.reshape(evaluated_x_points, (-1, n_params)), return_cov=True)
                sigma = np.sqrt(np.diag(Sigma))

                neg_expected_improvements = self.negative_expected_improvement(evaluated_x_points, y, n_params)

                selected_point = [next_sample, self.gp.predict(np.reshape(next_sample, (-1, n_params)), return_cov=False)]

                neg_improvement_at_selected_point = self.negative_expected_improvement(next_sample, y, n_params)
                acquisition = [next_sample, neg_improvement_at_selected_point]

                plot_progress(i, evaluated_x_points, evaluated_y_points, X, y, y_posterior, sigma, selected_point,
                              neg_expected_improvements, acquisition)
            # -------------------------------------------------------------------------------------------- #

            if np.any(np.abs(next_sample - X) <= self.epsilon):
                next_sample = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

            x_list.append(next_sample)
            y_list.append(self.score_func(next_sample))

            X = np.atleast_2d(np.array(x_list))
            y = np.array(y_list)

        return X, y

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
        return -1 * ei

    def sample_next_hyperparameter(self, acquisition_function, y, n_restart):

        n_params = self.bounds.shape[0]
        best_x = None
        best_acquisition_value = 100.0

        for initial_value in np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(n_restart, n_params)):
            res = minimize(fun=acquisition_function,
                           x0=initial_value,
                           bounds=self.bounds,
                           method='L-BFGS-B',
                           args=(y, n_params))

            if res.fun < best_acquisition_value:
                best_acquisition_value = res.fun
                best_x = res.x

        return best_x


def plot_progress(iter_idx, evaluated_x_points, evaluated_y_points, X, y, y_posterior, sigma, selected_point,
                  neg_expected_improvements, acquisition):

    import matplotlib
    import matplotlib.pyplot as plt
    from plot_functions import adjustAxeProperties

    matplotlib.style.use('ggplot')

    evaluated_x_points = evaluated_x_points.ravel()
    y_posterior = y_posterior.ravel()

    FONTSIZE = 25
    plt.close('all')
    fig = plt.figure(figsize=(11, 16))  # horizontal, vertical
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

    # plot the expected improvement
    ax = plt.subplot(gs[1, 0])
    ax.plot(evaluated_x_points, neg_expected_improvements, linestyle='-', linewidth=1.5, color='k')
    ax.scatter(acquisition[0], acquisition[1], label='Selected', marker="^", s=180, color='r')
    adjustAxeProperties(ax, FONTSIZE, 0, FONTSIZE, 0)
    ax.set_xlabel('x', fontsize=FONTSIZE, labelpad=15)
    ax.set_ylabel('Neg. expected improvements', fontsize=FONTSIZE, labelpad=15)

    plt.tight_layout(pad=0, w_pad=1.0, h_pad=2.0)
    fig.suptitle("Iteration: {0:}".format(iter_idx), fontsize=1.05*FONTSIZE)
    plt.subplots_adjust(top=0.93)
    fname = str(iter_idx).zfill(2)
    plt.savefig("iteration_{0:}.png".format(fname), bbox_inches='tight')


