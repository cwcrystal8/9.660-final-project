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

    return best_model



def main(groups = True, complexity = True, print_info = True):
    g1_df, g2_df = get_groups()
    all_df = pd.concat([g1_df, g2_df])
    all_df.columns = list(all_df.columns)

    if groups:

        print("---- GROUP 1 ANALYSIS ----")
        evaluate_models(g1_df)

        print("\n")

        print("---- GROUP 2 ANALYSIS ----")
        evaluate_models(g2_df)

        print("\n")

        print("---- ALL DATA ANALYSIS ----")
        
        evaluate_models(all_df)

    if complexity:
        print("\n---------------------------------------")
        print("\n--------------COMPLEXITY---------------")
        print("\n---------------------------------------")

        for name, df in {"Group 1": g1_df, "Group 2": g2_df, "Both Groups": all_df}.items():
            print(f"---- {name} ----")

            # r_value_comp_1 = evaluate_models(df, coins_to_ignore = [3,4,5,6], print_info = print_info)
            # print(f"\tComplexity 1: {r_value_comp_1}")

            r_value_comp_2 = evaluate_models(df, coins_to_ignore = [1,2,5,6], print_info = print_info)
            print(f"\tComplexity 2: {r_value_comp_2}")

            # r_value_comp_3 = evaluate_models(df, coins_to_ignore = [1,2,3,4], print_info = print_info)
            # print(f"\tComplexity 3: {r_value_comp_3}")
            
            print()

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main(groups=False, complexity=True, print_info = True)