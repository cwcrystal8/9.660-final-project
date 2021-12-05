from scipy.stats import linregress

from coins.sequence_info.sequences import number_to_sequence


def coin_1_likelihood(sequence):
    return 1 / (2 ** len(sequence))

def coin_2_likelihood(sequence):
    num_heads = sequence.count("H")
    num_tails = sequence.count("T")

    return (0.8 ** num_heads) * (0.2 ** num_tails)

def coin_3_likelihood(sequence):
    probability = 0.99 if sequence[0] == "H" else 0.01
    for prev, next in zip(sequence, sequence[1:]):
        probability *= 0.99 if prev != next else 0.01
    return probability

def coin_4_likelihood(sequence):
    probability = 1
    for prev, next in zip(sequence, sequence[1:]):
        if prev == "T":
            probability *= 0.99 if next == "H" else 0.01
        else:
            probability *= 0.5
    return probability

def coin_5_likelihood(sequence):
    probability = 0.99 if sequence[0] == "H" and sequence[1] == "T" else 0.01
    for prev_prev, prev, next in zip(sequence, sequence[1:], sequence[2:]):
        if prev_prev == "H" and prev == "T":
            probability *= 0.99 if next == "T" else 0.01
        elif prev_prev == "T" and prev == "T":
            probability *= 0.99 if next == "H" else 0.01
        elif prev_prev == "T" and prev == "H":
            probability *= 0.99 if next == "T" else 0.01
        else:
            probability *= 0.99 if next == "T" else 0.01
    return probability

def coin_6_likelihood(sequence):
    probability = 0.25
    for prev_prev, prev, next in zip(sequence, sequence[1:], sequence[2:]):
        if prev_prev == "T" and prev == "T":
            probability *= 0.99 if next == "H" else 0.01
        else:
            probability *= 0.5
    return probability

def likelihood(coin, sequence):
    """
    coin: int
    sequence: str

    Calculates P(sequence|coin)
    """
    sequence = sequence.split()
    mapping = {
        1: coin_1_likelihood,
        2: coin_2_likelihood,
        3: coin_3_likelihood,
        4: coin_4_likelihood,
        5: coin_5_likelihood,
        6: coin_6_likelihood

    }
    return mapping[coin](sequence)


def likelihood_model(gamma, g_df, coins_to_ignore = []):
    # Make each data point a row
    data_df = g_df.stack().reset_index().rename(columns = {"level_1": "(c,s)", 0: "Rating"})
    
    # Calculate likelihood
    data_df["Likelihood"] = data_df["(c,s)"].apply(lambda x: likelihood(x[0], number_to_sequence[x[1]]))

    # Map likelihood to the same interval
    data_df["Likelihood"] = 1 + data_df["Likelihood"] * 6

    # Get metadata for filtering
    data_df["Coin"] = data_df["(c,s)"].apply(lambda x: x[0])
    data_df["Sequence"] = data_df["(c,s)"].apply(lambda x: x[1])
    
    # Transform by power law
    data_df["Rating"] = data_df["Rating"] ** gamma
    
    # Filter out coins to ignore
    data_df = data_df[~(data_df["Coin"].isin(coins_to_ignore))]

    # Return r-value
    return linregress(data_df["Likelihood"], data_df["Rating"]).rvalue