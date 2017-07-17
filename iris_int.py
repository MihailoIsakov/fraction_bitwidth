import numpy as np

from activations import relu, relu_derivative
from sklearn.datasets import load_iris


def test(fraction, lr, weight_mean=0, weight_variance=500, iter=10000, func=relu, func_der=relu_derivative, verbosity="low"):
    np.random.seed(0xdeadbeec)

    iris = load_iris()

    x = iris.data
    # one hot vector
    y = np.zeros((len(x), 3))
    y[range(len(x)), iris.target] = 1

    # normalize
    x /= np.max(x, axis=0)

    # shift right by fraction size
    x *= 2**fraction
    y *= 2**fraction

    x = x.astype(int)
    y = y.astype(int)

    shuffle = np.random.permutation(np.arange(len(x)))
    x = x[shuffle]
    y = y[shuffle]

    MAX_SAMPLES = len(x)

    w = np.random.normal(0, weight_variance, (3, 4))
    w = np.rint(w).astype(int)

    accuracy = []
    classifications = np.zeros(1000)

    for i in range(iter): 
        sample = i % MAX_SAMPLES
        z0 = x[sample]
        
        target = y[sample]

        z1 = np.dot(w, z0)
        a1 = func(z1)
        z1 = np.right_shift(z1, fraction)
        a1 = np.right_shift(a1, fraction)

        delta = (target - a1) * func_der(z1)
        delta = np.round(delta).astype(int)

        # updates = z0.reshape(len(z0), 1) * delta.reshape(1, len(delta))
        updates = delta.reshape(len(delta), 1) * z0.reshape(1, len(z0))
        updates = np.right_shift(updates, fraction+lr)
        # updates = updates / 1000

        w = w + updates

        classifications[i % 1000] = np.argmax(a1) == np.argmax(target)
        accuracy.append(np.mean(np.argmax(a1) == np.argmax(target)))

        if verbosity == "low":
            print("accuracy: {}".format(np.mean(classifications)))

    return np.mean(classifications), accuracy


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
