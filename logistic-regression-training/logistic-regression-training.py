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
    X = np.array(X)  # shape (num_samples, num_features)
    y = np.array(y)  # shape (num_samples)
    num_training_samples = len(X)
    num_features = len(X[0])

    weights = np.zeros(num_features)  # shape (num_features)
    bias = 0.0

    for step in range(steps):
        predictions = _sigmoid(np.matmul(X, weights) + bias)

        # model loss at this step
        # loss = -1 / num_training_samples * sum(
        #     [
        #         gold_label * np.log(prediction) + (1 - gold_label) * np.log(1 - prediction)
        #         for prediction, gold_label in zip(predictions, y)
        #     ]
        # )

        weights_gradient = np.matmul(
            X.transpose(),
            (predictions - y),
        ) / num_training_samples
        biases_gradient = np.mean(predictions - y)

        # adjust weights
        weights = weights - lr * weights_gradient
        bias = bias - lr * biases_gradient

    return weights, bias
