import math


def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    prob_distributions_of_predicted_tokens = [
        prob_distributions[i][token]
        for i, token in enumerate(actual_tokens)
    ]
    neg_log_probability = - (1 / len(actual_tokens)) * sum(
        [
            math.log(prob) 
            for prob in prob_distributions_of_predicted_tokens
        ]
    )
    return math.exp(neg_log_probability)
