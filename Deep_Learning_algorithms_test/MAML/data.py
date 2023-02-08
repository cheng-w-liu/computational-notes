import numpy as np
import torch

def generate_one_sinusoidal_dataset(xs, amplitude, phase, K):
    sample_xs = np.random.choice(xs, size=K, replace=False)
    sample_ys = amplitude * np.sin(sample_xs - phase)
    return sample_xs, sample_ys

def generate_one_batch(support_size, query_size, batch_size):
    support_x_list, support_y_list, query_x_list, query_y_list = [], [], [] , []
    xs = np.linspace(-5, 5, 10000)
    true_data = []
    for b in range(batch_size):
        amplitude = np.random.uniform(0.1, 5.0)
        phase = np.random.uniform(0, np.pi)
        true_data.append((amplitude, phase))

        sample_xs, sample_ys = generate_one_sinusoidal_dataset(xs, amplitude, phase, support_size + query_size)
        support_x_list.append(torch.tensor(sample_xs[:support_size].reshape((support_size, 1))).float())
        support_y_list.append(torch.tensor(sample_ys[:support_size]).float())
        query_x_list.append(torch.tensor(sample_xs[support_size:].reshape((query_size, 1))).float())
        query_y_list.append(torch.tensor(sample_ys[support_size:]).float())

    return support_x_list, support_y_list, query_x_list, query_y_list, true_data