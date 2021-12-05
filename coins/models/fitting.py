import numpy as np

from coins.models.likelihood import likelihood_model
from coins.models.bayesian import bayesian_model
from coins.models.similarity import similarity_model, FEATURES

def row_search(model, df, min_gamma, max_gamma, delta, coins_to_ignore = []):
    best_r_value = - float("inf")
    best_gamma = None
    for gamma in range(int(min_gamma / delta), int(max_gamma / delta)):
        gamma *= delta
        r_value = model(gamma, df, coins_to_ignore = coins_to_ignore)
        if r_value > best_r_value:
            best_r_value = r_value
            best_gamma = gamma 
    return best_gamma

def fit_likelihood(df, coins_to_ignore = []):
    delta = 1
    min_gamma = 0
    max_gamma = 20
    best_gamma = None
    for i in range(3):
        best_gamma = row_search(likelihood_model, df, min_gamma, max_gamma, delta, coins_to_ignore=coins_to_ignore)
        is_first = best_gamma == min_gamma 
        is_last = best_gamma == max_gamma - delta 

        while is_first:
            diff = max_gamma - min_gamma
            min_gamma = best_gamma - diff // 2
            max_gamma = best_gamma + diff // 2

            best_gamma = row_search(likelihood_model, df, min_gamma, max_gamma, delta, coins_to_ignore=coins_to_ignore)
            is_first = best_gamma == min_gamma 

        while is_last:
            diff = max_gamma - min_gamma
            min_gamma = best_gamma - diff // 2
            max_gamma = best_gamma + diff // 2

            best_gamma = row_search(likelihood_model, df, min_gamma, max_gamma, delta, coins_to_ignore=coins_to_ignore)
            is_first = best_gamma == min_gamma 
            is_last = best_gamma == max_gamma - delta 
        
        min_gamma = best_gamma - delta 
        max_gamma = best_gamma + delta
        delta *= 0.1
    
    best_r_value = likelihood_model(best_gamma, df, coins_to_ignore = coins_to_ignore)
    return best_gamma, best_r_value

def fit_bayesian(df, coins_to_ignore = []):
    """
    alpha range: {0, 0.2, 0.4, ..., 1.8, 2.0}
    beta range:  {-1, -0.9, -0.8, ..., 0.9, 1}
    gamma range: {0, 0.4, 0.8, 1.2, 1.6, 2.0}
    """
    max_r = None
    max_alpha, max_beta, max_gamma = None, None, None
    for alpha in range(20):
        alpha = alpha / 10

        for beta in range(-10, 10):
            beta = beta / 10

            for gamma in range(5):
                gamma /= 2.5

                r = bayesian_model(alpha, beta, gamma, df, coins_to_ignore = coins_to_ignore)
                if max_r is None or r > max_r:
                    max_r = r
                    max_alpha, max_beta, max_gamma = alpha, beta, gamma
    return max_alpha, max_beta, max_gamma, max_r

def generate_feature_weights(n, prev):
    if n == 0:
        yield np.array(prev)
    else:
        for weight in [0.5 , 1, 1.5, 2]:
            yield from generate_feature_weights(n - 1, prev + [weight])

def fit_similarity(df, coins_to_ignore = []):
    best_feature_weights = []
    best_r_value = - float("inf")
    for feature_weights in generate_feature_weights(len(FEATURES), []):
        r_value = similarity_model(feature_weights, df, coins_to_ignore = coins_to_ignore)
        if r_value > best_r_value:
            best_r_value = r_value
            best_feature_weights = feature_weights
    return best_feature_weights, best_r_value

if __name__ == "__main__":
    from coins.data_cleaning.groups import get_groups
    g1_df, g2_df = get_groups()
    print(fit_similarity(g1_df))