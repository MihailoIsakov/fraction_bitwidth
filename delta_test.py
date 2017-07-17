"""
Script for testing delta means and variances.
Main goal is to figure out why networks with 12 or 20 bit fractions train, 
but networks with 16 bit fractions dont.
"""

import numpy as np

from activations import linear, relu, linear_derivative, relu_derivative
from backprop import forward, signed_shift, top_delta
from sklearn.datasets import fetch_mldata

np.random.seed(0xdeadbeec)


def test(fraction, lr, weight_variance=500, iter=10000, func=relu, func_der=relu_derivative, verbosity="low"):

    mnist = fetch_mldata('MNIST original')

    x = (mnist['data'] * 2**(fraction-8)).astype(int)
    y = np.zeros((len(x), 10))
    y[np.arange(len(x)).astype(int), mnist['target'].astype(int)] = 2**fraction

    shuffle = np.random.permutation(np.arange(len(x)))
    x = x[shuffle]
    y = y[shuffle].astype(int)

    MAX_SAMPLES = len(x)

    w = np.random.normal(0, weight_variance, (10, 784))
    w = np.rint(w).astype(int)

    errors        = np.zeros(iter)
    delta_means   = np.zeros(iter)
    delta_vars    = np.zeros(iter)
    update_errors = np.zeros(iter)

    for i in range(iter): 
        sample = i % MAX_SAMPLES
        z0 = x[sample]
        
        target = y[sample]

        z1, a1 = forward(z0, w, func)
        z1 = signed_shift(z1, fraction)
        a1 = signed_shift(a1, fraction)

        error = np.sum(np.abs(target - a1))

        delta = top_delta(target, a1, z1, func_der)

        # updates = z0.reshape(len(z0), 1) * delta.reshape(1, len(delta))
        updates = delta.reshape(len(delta), 1) * z0.reshape(1, len(z0))
        updates_shifted = signed_shift(updates, fraction+lr)

        # update mean and variance for this sample
        errors[i]       = error
        delta_means[i]  = np.mean(updates_shifted)
        delta_vars[i]   = np.var(updates_shifted)
        update_errors[i] = np.mean(np.abs(updates - updates_shifted*2**(fraction+lr)))

        w = w + updates_shifted

    return errors, delta_means, delta_vars


def main(iter=10000):
    results = []

    for lr in [0, 3, 6, 9, 12]:
        for fraction in [10, 12, 14, 16, 18, 20, 22]:
            print("Running fraction {} with LR {}".format(fraction, lr))
            errors, means, vars = test(fraction, lr, iter=iter, func=relu, func_der=relu_derivative)
            results.append((lr, fraction, errors, means, vars))

    return results


if __name__ == "__main__":
    main()
