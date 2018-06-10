import numpy as np


class RBFKernel(object):
    """
    Radial-basis function kernel:
    k(x, y) = exp(- 0.5 * ||x-y||^2 / l^2)
    where l = length_scale
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-3, 1e3), fixed_param=False):

        self.length_scale = np.array(length_scale)
        self.fixed_param = fixed_param
        if not fixed_param:
            assert np.iterable(length_scale_bounds)
        if np.iterable(length_scale):
            self.n_elements = len(length_scale)
            length_scale_bounds = np.atleast_2d(length_scale_bounds)
            if length_scale_bounds.shape[0] == 1:
                self.length_scale_bounds = np.repeat(length_scale_bounds, self.n_elements, 0)
            elif length_scale_bounds.shape[0] != self.n_elements:
                raise Exception("dimension mismatched. length_scale: {0:}, bounds: {1:}".format(len(self.length_scale), length_scale_bounds.shape))
            else:
                self.length_scale_bounds = np.atleast_2d(length_scale_bounds)
        else:
            self.n_elements = 1
            self.length_scale_bounds = np.atleast_2d(length_scale_bounds)

    @property
    def n_dims(self):
        return self.n_elements

    def clone_with_param(self, new_length_scale):
        if np.iterable(self.length_scale):
            if not np.iterable(new_length_scale):
                raise Exception("new_legnth_scale is not iterable")
            if len(self.length_scale) != len(new_length_scale):
                raise Exception("new_length_scale mismatched")
        return RBFKernel(new_length_scale, self.length_scale_bounds)

    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    def get_parameters(self):
        if self.fixed_param:
            return {'length_scale': self.length_scale}
        else:
            return {'length_scale': self.length_scale, 'bounds': self.length_scale_bounds}

    def get_length_scale(self):
        return self.length_scale

    def set_length_scale(self, new_length_scale):
        self.length_scale = new_length_scale

    def __call__(self, X, Y=None, evaluate_gradient=False):
        """
        return the evaluated kernel values: k(X, Y), and 
        optionally the gradient w.r.t. scale_length evaluated at (X, Y)
        """
        if Y is None:
            K = self.k_(X, X)
        else:
            if evaluate_gradient:
                raise Exception("gradient can only evaluated when Y is None")
            K = self.k_(X, Y)

        if evaluate_gradient:
            if self.fixed_param:
                return K, np.empty((len(X), len(X)))
            else:
                if self.anisotropic():
                    N = X.shape[0]
                    K_grad = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2
                    K_grad /= np.tile(self.length_scale ** 3, (N, N, 1))
                    K_grad *= K[:, :, np.newaxis]
                else:
                    K_grad = K * self._euclidean_distance_square(X, X) / np.power(self.length_scale, 3)
                    K_grad = K_grad[:, :, np.newaxis]
                return K, K_grad
        else:
            return K

    def k_(self, X, Y=None):
        if len(X.shape) != 2:
            raise Exception("X should be a matrix")
        if Y is not None and len(Y.shape) != 2:
            raise Exception("Y should be a matrix")

        if Y is None:
            length_scale = np.tile(self.length_scale, (X.shape[0], 1))
            euclidean_distance_squared = self._euclidean_distance_square(X / length_scale, X / length_scale)
        else:
            length_scale_x = np.tile(self.length_scale, (X.shape[0], 1))
            length_scale_y = np.tile(self.length_scale, (Y.shape[0], 1))
            euclidean_distance_squared = self._euclidean_distance_square(X / length_scale_x, Y / length_scale_y)
        similarity = np.exp(-0.5 * euclidean_distance_squared)
        return similarity
    
    def _euclidean_distance_square(self, X, Y):
        """
        calculate ||X-Y||^2, where each row of X (and Y) represents a data point
        """
        XX = self._row_norm_square(X)
        YY = self._row_norm_square(Y)
        XY = np.dot(X, Y.T)
        # remember to transpose YY so that the sum of matrices will use broadcast
        d = XX + YY.T - 2 * XY
        return d

    def _row_norm_square(self, X):
        return np.sum(X*X, axis=1, keepdims=True)



if __name__ == '__main__':
    from scipy.optimize import check_grad
    from sklearn.utils.testing import assert_almost_equal

    print('running gradient test')

    d = np.random.randint(1, 10)
    print("data points in {0:}-dimensional".format(d))

    X = np.random.normal(size=(13, d))

    def func(theta, i, j):
        K = RBFKernel(theta)(X)
        return K[i, j]

    def grad(theta, i, j):
        K = RBFKernel(theta)(X, evaluate_gradient=True)[1]
        return K[i, j, :]

    # test diagonal elements
    for _ in range(20):
        theta_value = np.random.normal(size=d)
        K_grad = RBFKernel(theta_value)(X, evaluate_gradient=True)[1]
        for k in range(d):
            for i in range(X.shape[0]):
                assert K_grad[i, i, k] == 0.0

    # test off-diagonal elements
    errors = []
    for _ in range(20):
        theta_value = np.random.normal(size=d)
        for row_idx in range(X.shape[0]):
            for col_idx in range(row_idx+1, X.shape[0]):
                err = check_grad(func, grad, theta_value, row_idx, col_idx)
                if err > 0.0:
                    errors.append(err)
                err = check_grad(func, grad, theta_value, col_idx, row_idx)
                if err > 0.0:
                    errors.append(err)
    assert_almost_equal(errors, np.zeros_like(errors), 4)

