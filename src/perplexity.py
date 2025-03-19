
import numpy as np

def perplexity(probabilities):
    """
    Compute perplexity given an array of token probabilities.

    :param probabilities: List or NumPy array of token probabilities.
    :return: Perplexity score.
    """
    probabilities = np.array(probabilities)
    
    # Avoid log(0) by replacing zeros with a small epsilon
    epsilon = 1e-10
    probabilities = np.clip(probabilities, epsilon, 1.0)
    
    # Compute log probabilities and average them
    log_probs = np.log(probabilities)
    avg_log_prob = np.mean(log_probs)
    
    # Compute perplexity
    return np.exp(-avg_log_prob)

# # Example Usage:
# token_probs = [0.1, 0.5, 0.2, 0.2]  # Example token probabilities
# pp = perplexity(token_probs)
# print("Perplexity:", pp)

