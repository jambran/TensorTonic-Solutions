import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    if sum(p) != 1:
        raise ValueError('Invalid probability distribution. Probabilities must sum to 1.')
    return sum(
        x_i * p_i 
        for (x_i, p_i) in zip(x, p)
    )
