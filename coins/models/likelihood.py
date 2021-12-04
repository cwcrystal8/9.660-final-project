from coins.coin_info.validation import validation

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
        3: coin_3_likelihood

    }
    return mapping[coin](sequence)
