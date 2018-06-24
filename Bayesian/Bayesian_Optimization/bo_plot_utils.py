import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from plot_functions import adjustAxeProperties
matplotlib.style.use('ggplot')


def plot_1d(iter_idx, evaluated_x_points, X, y, y_posterior, sigma, selected_point,
                  acquisitions, acquired_point, prefix=''):

    evaluated_x_points = evaluated_x_points.ravel()
    y_posterior = y_posterior.ravel()

    FONTSIZE = 23
    plt.close('all')
    fig = plt.figure(figsize=(12, 16))  # horizontal, vertical
    gs = matplotlib.gridspec.GridSpec(2, 1)  # vertical, horizontal

    # plot function, GP posterior, and the optimal point
    ax = plt.subplot(gs[0, 0])
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



def plot_2d(iter_idx, param1_grid, param2_grid, X_history, mu, selected_point,
                  acquisitions, param1_text='', param2_text='', prefix=''):

    FONTSIZE = 23

    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10)) #, sharex=True, sharey=True)
    X, Y = np.meshgrid(param1_grid, param2_grid, indexing='ij')

    cp1 = ax1.contourf(X, Y, mu.reshape(X.shape))
    plt.colorbar(cp1, ax=ax1)
    ax1.autoscale(False)
    ax1.scatter(X_history[:, 0], X_history[:, 1], zorder=1, s=80, alpha=0.6)
    ax1.axvline(selected_point[0], color='k')
    ax1.axhline(selected_point[1], color='k')
    ax1.scatter(selected_point[0], selected_point[1], marker='x', color='r', s=100)
    ax1.set_title("Posterior function values", fontsize=FONTSIZE)
    ax1.set_xlabel("Param1 " + param1_text, fontsize=FONTSIZE, labelpad=15)
    ax1.set_ylabel("Param2 " + param2_text, fontsize=FONTSIZE, labelpad=15)
    adjustAxeProperties(ax1, FONTSIZE, 0, FONTSIZE, 0)

    cp2 = ax2.contourf(X, Y, np.reshape(acquisitions, X.shape))
    plt.colorbar(cp2, ax=ax2)
    ax2.autoscale(False)
    ax2.axvline(selected_point[0], color='k', ls='--', lw=1.5)
    ax2.axhline(selected_point[1], color='k', ls='--', lw=1.5)
    ax2.scatter(selected_point[0], selected_point[1], marker='x', color='r', s=100)
    ax2.set_title("Acquisition function values", fontsize=FONTSIZE)
    ax2.set_xlabel("Param1 " + param1_text, fontsize=FONTSIZE, labelpad=15)
    ax2.set_ylabel("Param2 " + param2_text, fontsize=FONTSIZE, labelpad=15)
    adjustAxeProperties(ax2, FONTSIZE, 0, FONTSIZE, 0)

    plt.tight_layout(pad=0, w_pad=1.0, h_pad=4.0)
    fig.suptitle("Iteration: {0:}".format(iter_idx), fontsize=FONTSIZE)
    plt.subplots_adjust(top=0.89)
    fname = str(iter_idx).zfill(2)
    plt.savefig(prefix + "iteration_{0:}.png".format(fname), bbox_inches='tight')


def plot_iteration(bo_obj, param1_grid, X_history, y_history, n_pre_samples, param2_grid,
                   param1_text='', param2_text=''):
    bo_model = bo_obj.clone()
    policy = bo_model.policy
    n_samples, n_params = X_history.shape
    for i in range(n_pre_samples, n_samples-1):
        bo_model.gp.fit(X_history[0:(i+1)], y_history[0:(i+1)])
        if param2_grid is None:
            mu, Sigma = bo_model.gp.predict(np.reshape(param1_grid, (-1, 1)), return_cov=True)
            sigma = np.sqrt(np.diag(Sigma))
            selected_point = (X_history[i+1, :], y_history[i+1])
            acquisitions = bo_model.acquisition_function(param1_grid, y_history[0:(i+1)], 1, policy)
            acquired_point = (X_history[i+1, :],
                              bo_model.acquisition_function(X_history[i+1, :], y_history[0:(i+1)], 1, policy))
            plot_1d(i, param1_grid, X_history[0:(i+1)], y_history[0:(i+1)], mu, sigma, selected_point,
                    acquisitions, acquired_point, prefix='')
        else:
            params_grid = np.array([[param1, param2] for param1 in param1_grid for param2 in param2_grid])
            mu = bo_model.gp.predict(params_grid, return_cov=False)
            acquisitions = bo_model.acquisition_function(params_grid, y_history[0:(i+1)], 2, policy)
            plot_2d(i, param1_grid, param2_grid, X_history, mu, X_history[i+1, :], acquisitions,
                    param1_text, param2_text)
