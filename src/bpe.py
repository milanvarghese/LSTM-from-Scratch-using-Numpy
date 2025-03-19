import numpy as np
from collections import Counter

def get_pair_frequencies(tokens):
    """Compute frequencies of adjacent token pairs."""
    pairs = [tuple(tokens[i:i+2]) for i in range(len(tokens) - 1)]
    return Counter(pairs)

def merge_pair(tokens, pair, new_token):
    """Merge the most frequent pair in the token list."""
    merged_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
            merged_tokens.append(new_token)
            i += 2  # Skip next token as it's merged
        else:
            merged_tokens.append(tokens[i])
            i += 1
    return merged_tokens

def byte_pair_encoding(text, num_merges):
    """Train BPE and return tokens, vocab, and learned merges."""
    tokens = list(text)  # Initialize as character tokens
    vocab = set(tokens)  # Initial vocabulary
    merges = {}  # Store merge rules

    for _ in range(num_merges):
        pair_freqs = get_pair_frequencies(tokens)
        if not pair_freqs:
            break
        most_frequent_pair = max(pair_freqs, key=pair_freqs.get)

        new_token = "".join(most_frequent_pair)  # Merge pair into a new token
        vocab.add(new_token)  # Add new token to vocabulary
        merges[most_frequent_pair] = new_token  # Store merge step

        tokens = merge_pair(tokens, most_frequent_pair, new_token)

    return tokens, vocab, merges

def tokenize_with_bpe(text, merges):
    """Tokenizes text using learned BPE merges."""
    tokens = list(text)  # Start with character-level tokens

    # Apply stored merges in order
    while True:
        pair_freqs = get_pair_frequencies(tokens)
        if not pair_freqs:
            break
        
        merge_candidates = [(pair, merges[pair]) for pair in pair_freqs if pair in merges]
        if not merge_candidates:
            break

        # Apply the first merge in stored order
        pair_to_merge, new_token = merge_candidates[0]
        tokens = merge_pair(tokens, pair_to_merge, new_token)

    return tokens


