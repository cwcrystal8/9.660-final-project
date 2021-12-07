import pandas as pd

from coins.data_cleaning.groups import get_groups
from coins.models.fitting import fit_likelihood, fit_bayesian, fit_similarity

def evaluate_models(df, coins_to_ignore = [], print_info = True):
    
    gamma, r_value_l = fit_likelihood(df, coins_to_ignore = coins_to_ignore)
    if print_info:
        print("Likelihood model:")
        print(f"\tgamma = {gamma}")
        print(f"\tr-value = {r_value_l}")

    
    alpha, beta, gamma, r_value_b = fit_bayesian(df, coins_to_ignore = coins_to_ignore)
    if print_info:
        print()
        print("Bayesian model:")
        print(f"\talpha = {alpha}")
        print(f"\tbeta = {beta}")
        print(f"\tgamma = {gamma}")
        print(f"\tr-value = {r_value_b}")  

    
    feature_weights, r_value_s = fit_similarity(df, coins_to_ignore = coins_to_ignore)
    if print_info:
        print()
        print("Similarity model:")
        print(f"\tfeature weights = {feature_weights}")
        print(f"\tr-value = {r_value_s}")

    
    best_model = max([
        (r_value_l, "Likelihood Model"),  
        (r_value_b, "Bayesian Model"),
        (r_value_s, "Similarity Model"),
    ])
    if print_info:
        print()
        print(f"Best Model: {best_model[1]}")

    return best_model[0]



def main():
    g1_df, g2_df = get_groups()

    print("---- GROUP 1 ANALYSIS ----")
    evaluate_models(g1_df)

    print("\n")

    print("---- GROUP 2 ANALYSIS ----")
    evaluate_models(g2_df)

    print("\n")

    print("---- ALL DATA ANALYSIS ----")
    df = pd.concat([g1_df, g2_df])
    df.columns = list(df.columns)
    evaluate_models(df)

    print("\n---------------------------------------\n")

    r_value_comp_1 = evaluate_models(df, coins_to_ignore = [3,4,5,6], print_info = False)
    print(f"\tComplexity 1: r-value = {r_value_comp_1}")

    r_value_comp_2 = evaluate_models(df, coins_to_ignore = [1,2,5,6], print_info = False)
    print(f"\tComplexity 2: r-value = {r_value_comp_2}")

    r_value_comp_3 = evaluate_models(df, coins_to_ignore = [1,2,3,4], print_info = False)
    print(f"\tComplexity 3: r-value = {r_value_comp_3}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()