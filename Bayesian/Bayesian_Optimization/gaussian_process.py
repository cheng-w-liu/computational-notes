from kernels import RBFKernel
import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import fmin_l_bfgs_b
import operator


class GaussianProcess(object):

    def __init__(self, kernel, alpha=1e-6, optimizer="fmin_l_bfgs_b", n_restart_optimizer=10):
        self.kernel_ = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restart_optimizer = n_restart_optimizer

    def fit(self, X, y):

        if len(X.shape) != 2:
            raise Exception("X needs to be 2D")
        self.X_train_ = np.copy(X)
        self.y_train_ = np.copy(y)

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            optima = []

            def obj_func(theta, evaluate_gradient=True):
                if evaluate_gradient:
                    lml, lml_grad = self.log_marginal_likelihood(theta, evaluate_gradient=True)
                    return -lml, -lml_grad
                else:
                    lml = self.log_marginal_likelihood(theta, evaluate_gradient=False)
                    return -lml

            bounds = self.kernel_.get_parameters()['bounds']
            optima.append(self._constrained_optimization(obj_func,
                                                         self.kernel_.get_parameters()['length_scale'],
                                                         bounds))

            if self.n_restart_optimizer > 0:
                for _ in range(self.n_restart_optimizer):
                    initial_theta = np.random.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, initial_theta, bounds))

            lml_values = list(map(operator.itemgetter(1), optima))
            self.log_marginal_likelihood_value = np.max(lml_values)
            self.kernel_.set_length_scale(optima[np.argmax(lml_values)][0])
        else:
            # no need to optimize the hyperparameter, use the values as provided
            self.log_marginal_likelihood_value = self.log_marginal_likelihood(self.kernel_.get_parameters()['length_scale'])


        # pre-compute the relevant quantities that will be used in `predict`
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            print("K is not positive definite. Try increase alpha")
            raise
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)

    def predict(self, X, return_cov=False):
        Ks = self.kernel_(self.X_train_, X)

        y_mean = np.dot(Ks.T, self.alpha_)

        if return_cov:
            Kss = self.kernel_(X, X)
            v = cho_solve((self.L_, True), Ks)
            y_Cov = Kss - np.dot(Ks.T, v)
            return y_mean, y_Cov
        else:
            return y_mean

    def log_marginal_likelihood(self, theta, evaluate_gradient=False):
        """
        - The log marginal likelihood can be written as:
          log P(y|X) = - 1/2 y^{T} K^{-1} y - 1/2 log det(K) - N/2 log(2 pi)
                   = -1/2 y^{T} alpha - \sum_i \log L_ii  - N/2 log(2 pi)
         where L is the lower triangular matrix of the Cholesky decomposition of K

        - The gradient of the log marginal likelihood can be written as:
         d (log p(y|X)) / d theta_j = 1/2 Tr [(alpha alpha^T - K^{-1}) dK / d theta_j]
        """
        kernel = self.kernel_.clone_with_param(theta)
        if evaluate_gradient:
            K, K_grad = kernel(self.X_train_, evaluate_gradient=True)
        else:
            K = kernel(self.X_train_, evaluate_gradient=False)
        K[np.diag_indices_from(K)] += self.alpha

        try:
            L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            if evaluate_gradient:
                return (-np.inf, np.zeros_like(theta))
            else:
                return -np.inf

        alpha_ = cho_solve((L, True), self.y_train_)

        lml = -0.5 * np.dot(self.y_train_.T, alpha_)
        lml += -np.sum(np.log(np.diag(L)))
        lml += -0.5 * K.shape[0] * np.log(2.0 * np.pi)

        if evaluate_gradient:
            aa = np.outer(alpha_, alpha_)
            K_inv = cho_solve((L, True), np.eye(K.shape[0]))
            M = aa - K_inv
            lml_grad = np.zeros(self.kernel_.n_dims)
            for k in range(self.kernel_.n_dims):
                lml_grad[k] = 0.5 * np.trace(np.dot(M, K_grad[:, :, k]))

        if evaluate_gradient:
            return lml, lml_grad
        else:
            return lml

    def _constrained_optimization(self, obj_func, initial_value, bounds):

        if self.optimizer == 'fmin_l_bfgs_b':
            theta_opt, fmin, convergence_info = \
              fmin_l_bfgs_b(obj_func, initial_value, bounds=bounds)
            if convergence_info["warnflag"] != 0:
                print("fmin_l_bfgs_b terminated abnormally. State: {0:}".format(convergence_info))
        elif callable(self.optimizer):
            theta_opt, fmin = self.optimizer(obj_func, initial_value, bounds=bounds)
        else:
            raise ValueError("unknown optimizer: {0:}".format(self.optimizer))

        return theta_opt, -fmin


if __name__ == '__main__':
    from scipy.optimize import check_grad
    from sklearn.utils.testing import assert_almost_equal

    print('Test gradient of log marginal likelihood')

    np.random.seed(10)
    d = 6
    n_examples = 10

    X = np.random.normal(size=(n_examples, d))
    y = np.random.normal(size=n_examples)

    kernel = RBFKernel(np.arange(d)+1)
    gp = GaussianProcess(kernel)

    def func(theta):
        cloned_kernel = gp.kernel_.clone_with_param(theta)
        cloned_gp = GaussianProcess(cloned_kernel, optimizer=None)
        cloned_gp.fit(X, y)
        lml = cloned_gp.log_marginal_likelihood_value
        return lml

    def grad(theta):
        cloned_kernel = gp.kernel_.clone_with_param(theta)
        cloned_gp = GaussianProcess(cloned_kernel, optimizer=None)
        cloned_gp.fit(X, y)
        lml, lml_grad = cloned_gp.log_marginal_likelihood(theta, evaluate_gradient=True)
        return lml_grad

    for _ in range(10):
        theta_value = np.random.normal(size=d)
        err = check_grad(func, grad, theta_value)
        try:
            assert_almost_equal(err, 0.0, 4)
        except:
            print("failed on : {0:}".format(theta_value))