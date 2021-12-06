from math import log, e
from scipy.stats import linregress

from coins.models.likelihood import likelihood
from coins.sequence_info.sequences import number_to_sequence

def bayesian(coin, sequence, all_coins = (1,2,3,4,5,6)):
    numerator = likelihood(coin, sequence)
    
    denominator = 0
    n = len(all_coins)
    for other_coin in all_coins:
        if other_coin != coin:
            p_hj_hi = 1 / (n - 1) # assuming each hypothesis is equally likely
            denominator += likelihood(other_coin, sequence) * p_hj_hi 
    
    return log(numerator / denominator)

def bayesian_model(alpha, beta, gamma, g_df, coins_to_ignore = []):
    # Make each data point a row
    data_df = g_df.stack().reset_index().rename(columns = {"level_1": "(c,s)", 0: "Rating"})
    
    # Calculate bayesian
    data_df["Bayesian"] = data_df["(c,s)"].apply(lambda x: bayesian(x[0], number_to_sequence[x[1]]))

    # Transform by power law
    data_df["Rating"] = data_df["Rating"] ** gamma

    # Map bayesian
    data_df["Bayesian"] = -alpha * data_df["Bayesian"] + beta
    data_df["Bayesian"] = 1 / (1 + e ** data_df["Bayesian"])
    data_df["Bayesian"] = 1 + 6 * data_df["Bayesian"]
    
    # Get metadata for filtering
    data_df["Coin"] = data_df["(c,s)"].apply(lambda x: x[0])
    data_df["Sequence"] = data_df["(c,s)"].apply(lambda x: x[1])
    
    # Filter out coins to ignore
    data_df = data_df[~(data_df["Coin"].isin(coins_to_ignore))]
    
    # Return r-value
    return linregress(data_df["Bayesian"], data_df["Rating"]).rvalue