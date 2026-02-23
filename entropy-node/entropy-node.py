import numpy as np
from collections import Counter
import math

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    if not y:
        return 0.0
    class_to_count = Counter(y)
    num_samples = len(y)
    return -sum(
        class_to_count[class_] / num_samples * math.log(
            class_to_count[class_] / num_samples,
            2,
        )
        for class_ in class_to_count.keys()
    )