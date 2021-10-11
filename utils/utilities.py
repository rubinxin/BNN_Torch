import numpy as np


def zero_one_normalization(X, lower=None, upper=None):

    if lower is None:
        lower = np.min(X, axis=0)
    if upper is None:
        upper = np.max(X, axis=0)

    X_normalized = np.true_divide((X - lower), (upper - lower))

    return X_normalized, lower, upper


def zero_one_denormalization(X_normalized, lower, upper):
    return lower + (upper - lower) * X_normalized


def zero_mean_unit_var_normalization(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    X_normalized = (X - mean) / std

    return X_normalized, mean, std


def zero_mean_unit_var_denormalization(X_normalized, mean, std):
    return X_normalized * std + mean


def iterate_minibatches(inputs, targets, batchsize, rng, shuffle=False):
    assert inputs.shape[0] == targets.shape[0], \
        "The number of training points is not the same"
    if shuffle:
        indices = np.arange(inputs.shape[0])
        rng.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



def  get_init_data(obj_func, noise_var, n_init, bounds, hetero=None):

    d = bounds.shape[0]
    x_init = np.random.uniform(bounds[0,0], bounds[0,1], (n_init, d))
    f_init = obj_func(x_init)
    if hetero is None:
        y_init = f_init + np.sqrt(noise_var) * np.random.randn(n_init, 1)
    else:
        y_init = f_init + (hetero(x_init)*np.sqrt(noise_var)) * np.random.randn(n_init, 1)

    return x_init, y_init