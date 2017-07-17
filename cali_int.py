import numpy as np

from activations import relu, relu_derivative
from sklearn.datasets import fetch_california_housing


def test(fraction, lr, weight_mean=0, weight_variance=500, iter=10000, classification_number=1000, func=relu, func_der=relu_derivative, verbosity="low"):
    np.random.seed(0xdeadbeec)

    cali = fetch_california_housing()
    x, y = cali['data'], cali['target']
    x /= np.max(x, axis=0)
    y /= np.max(y, axis=0)

    x *= 2**fraction
    y *= 2**fraction

    x = x.astype(int)
    y = y.astype(int)

    shuffle = np.random.permutation(np.arange(len(x)))
    x = x[shuffle]
    y = y[shuffle]

    # MAX_SAMPLES = len(x)
    MAX_SAMPLES = 1

    w0 = np.random.normal(0, weight_variance, (100, 8))
    w0 = np.rint(w0).astype(int)

    w1 = np.random.normal(0, weight_variance, (1, 100))
    w1 = np.rint(w1).astype(int)

    error_buffer = np.zeros(iter)

    for i in range(iter): 
        sample = i % MAX_SAMPLES
        z0     = x[sample]
        target = y[sample]

        z1 = np.matmul(w0, z0.reshape(len(z0), 1))
        a1 = func(z1)
        z1 = np.right_shift(z1, fraction)
        a1 = np.right_shift(a1, fraction)

        z2 = np.matmul(w1, z1.reshape(len(z1), 1))
        a2 = func(z2)
        z2 = np.right_shift(z2, fraction)
        a2 = np.right_shift(a2, fraction)

        delta2 = (target - a2) * func_der(z2)
        delta2 = np.round(delta2).astype(int)

        delta1 = np.matmul(w1.T, delta2) * func(z1)
        delta1 = np.round(delta1).astype(int)

        # updates = z0.reshape(len(z0), 1) * delta.reshape(1, len(delta))
        updates2 = delta2.reshape(len(delta2), 1) * z1.reshape(1, len(z1))
        updates2 = np.right_shift(updates2, fraction+lr)
        print updates2

        updates1 = delta1.reshape(len(delta1), 1) * z0.reshape(1, len(z0))
        updates1 = np.right_shift(updates1, fraction+lr)
        # updates = updates / 1000

        w1 = w1 + updates2
        w0 = w0 + updates1

        error_buffer[i] = np.sum(np.abs(target - a2))

        if verbosity == "low":
            print("error: {}".format(error_buffer[i] / 2**fraction))

    return error_buffer


def main():
    accuracy = []

    for lr in range(-3, 15):
        for fraction in range(5, 32):
            acc = test(fraction, lr, iter=1000000, func=relu, func_der=relu_derivative, verbosity="none")
            accuracy.append((lr, fraction, acc))

    return accuracy


if __name__ == "__main__":
    # test(20, 3, 100000, func=relu, func_der=relu_derivative, verbosity="low")
    main()
