from coins.coin_info.validation import validation
from coins.sequence_info.sequences import number_to_sequence
from scipy.stats import linregress

def coin_1_likelihood(sequence):
    return 1 / (2 ** len(sequence))

def coin_2_likelihood(sequence):
    num_heads = sequence.count("H")
    num_tails = sequence.count("T")

    return (0.8 ** num_heads) * (0.2 ** num_tails)

def coin_3_likelihood(sequence):
    return 0.99 if validation(3, sequence) else 0.01

def coin_4_likelihood(sequence):
    if validation(4, sequence):
        probability = 1
        for i in range(len(sequence)):
            if i < 1:
                flip_prob = 0.5
            else:
                flip_prob = 1 if sequence[i - 1] == "T" else 0.5
            probability *= flip_prob
        return 0.99 * probability
    return 0.01

def coin_5_likelihood(sequence):
    return 0.99 if validation(5, sequence) else 0.01

def coin_6_likelihood(sequence):
    if validation(6, sequence):
        probability = 1
        for i in range(len(sequence)):
            if i < 2:
                flip_prob = 0.5
            else:
                flip_prob = 1 if {sequence[i - 1], sequence[i-2]} == {"T"} else 0.5
            probability *= flip_prob
        return 0.99 * probability
    return 0.01

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


def test_likelihood(gamma, g_df, coins_to_ignore = []):
    # Make each data point a row
    data_df = g_df.stack().reset_index().rename(columns = {"level_1": "(c,s)", 0: "Rating"})
    
    # Calculate likelihood
    data_df["Likelihood"] = data_df["(c,s)"].apply(lambda x: likelihood(x[0], number_to_sequence[x[1]]))
    
    # Get metadata for filtering
    data_df["Coin"] = data_df["(c,s)"].apply(lambda x: x[0])
    data_df["Sequence"] = data_df["(c,s)"].apply(lambda x: x[1])
    
    # Transform by power law
    data_df["Rating"] = data_df["Rating"] ** gamma
    
    # Filter out coins to ignore
    data_df = data_df[~(data_df["Coin"].isin(coins_to_ignore))]

    # Return r-value
    return linregress(data_df["Likelihood"], data_df["Rating"]).rvalue