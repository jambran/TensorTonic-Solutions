import numpy as np


def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z)),
    )


def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y)

    num_training_samples, num_features = X.shape
    weights = np.zeros(num_features)
    bias = 0

    for step in range(steps):

        # inference
        probabilities = _sigmoid(
            np.matmul(X, weights) + bias
        )

        # update weights
        difference = probabilities - y
        gradient_w = np.matmul(X.transpose(), difference) / num_training_samples
        gradient_b = np.mean(difference)

        weights = weights - lr * gradient_w
        bias = bias - lr * gradient_b


    return weights, bias
