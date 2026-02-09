import numpy as np

from collections import Counter

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    token_to_index = {
        word: i for i, word in enumerate(vocab)
    }

    token_to_count = Counter(tokens)

    return np.array(
        [
            token_to_count[token_type] 
            for token_type in vocab
        ],
        dtype=int,
    )