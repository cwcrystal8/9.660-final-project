import numpy as np
from math import e
from scipy.stats import linregress

from coins.models.constants import expected_features_mapping
from coins.models.likelihood import likelihood
from coins.sequence_info.sequences import number_to_sequence


## Features

def num_heads(sequence):
    return sequence.count("H")

def num_h_t(sequence):
    return sum(1 if a != b else 0 for a, b in zip(sequence, sequence[1:]))

def num_h_after_t(sequence):
    return sum(1 if a != b and b == "H" else 0 for a, b in zip(sequence, sequence[1:]))

def num_h_t_t(sequence):
    return sum(1 if (a,b,c) == ("H", "T", "T") else 0 for a, b,c in zip(sequence, sequence[1:], sequence[2:]))

def num_h_after_t_t(sequence):
    return sum(1 if (a,b,c) == ("T", "T", "H") else 0 for a, b,c in zip(sequence, sequence[1:], sequence[2:]))

## Calculating the prototype for each hypothesis/feature/sequence
COINS = [1,2,3,4,5,6]
SEQUENCE_LENGTHS = [5, 13, 14, 10, 11, 6]
FEATURES = [num_heads, num_h_t, num_h_after_t, num_h_t_t, num_h_after_t_t]

def calc_expected(feature_func, coin, n, prev):
    if n == 0:
        return likelihood(coin, " ".join(prev)) * feature_func(prev)
    total = 0
    for flip in {"H", "T"}:
        curr = prev + [flip]
        new = calc_expected(feature_func, coin, n - 1, curr)
        total += new 
    return total

def make_mapping():
    mapping = {}
    print("mapping = {")
    for coin in COINS:
        for n in SEQUENCE_LENGTHS:
            for i, feature_func in enumerate(FEATURES):
                expected = calc_expected(feature_func, coin, n, [])
                mapping[(i, coin, n)] = expected

                print(f"\t{(i, coin, n)}: {expected},")
    print("}")

    return mapping

## Model

def get_sequence_feature_vector(sequence):
    return np.array([feature(sequence) for feature in FEATURES])

def get_coin_feature_vector(coin, n):
    return np.array([expected_features_mapping[(i, coin, n)] for i in range(len(FEATURES))])


def sim(feature_weights, d_i, d_j):
    return e ** (- feature_weights @ np.abs(d_i - d_j))


def similarity(feature_weights, coin, sequence, coins_to_ignore = []):
    sequence = sequence.split()
    
    coin_vector = get_coin_feature_vector(coin, len(sequence))
    sequence_vector = get_sequence_feature_vector(sequence)

    numerator = sim(feature_weights, coin_vector, sequence_vector)
    denominator = 0
    for other_coin in COINS:
        if other_coin not in coins_to_ignore:
            other_coin_vector = get_coin_feature_vector(other_coin, len(sequence))
            denominator += sim(feature_weights, other_coin_vector, sequence_vector)
    
    return numerator / denominator

def similarity_model(feature_weights, g_df, coins_to_ignore = []):
    # Make each data point a row
    data_df = g_df.stack().reset_index().rename(columns = {"level_1": "(c,s)", 0: "Rating"})
    
    # Calculate similarity
    data_df["Similarity"] = data_df["(c,s)"].apply(lambda x: similarity(feature_weights, x[0], number_to_sequence[x[1]], coins_to_ignore = coins_to_ignore))

    # Map similarity to the same interval
    data_df["Similarity"] = 1 + data_df["Similarity"] * 6

    # Get metadata for filtering
    data_df["Coin"] = data_df["(c,s)"].apply(lambda x: x[0])
    data_df["Sequence"] = data_df["(c,s)"].apply(lambda x: x[1])
    
    # Filter out coins to ignore
    data_df = data_df[~(data_df["Coin"].isin(coins_to_ignore))]

    # Return r-value
    return linregress(data_df["Similarity"], data_df["Rating"]).rvalue



if __name__ == "__main__":
    # make_mapping()

    from coins.data_cleaning.groups import get_groups
    g1_df, g2_df = get_groups()
    print(similarity_model(np.ones(len(FEATURES)), g1_df))


