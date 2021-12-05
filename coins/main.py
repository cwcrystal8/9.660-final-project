from coins.data_cleaning.groups import get_groups
from coins.models.fitting import fit_likelihood, fit_bayesian, fit_similarity

def evaluate_models(df):
    print("Likelihood model:")
    gamma, r_value_l = fit_likelihood(df)
    print(f"\tgamma = {gamma}")
    print(f"\tr-value = {r_value_l}")

    print()

    print("Bayesian model:")
    alpha, beta, gamma, r_value_b = fit_bayesian(df)
    print(f"\talpha = {alpha}")
    print(f"\tbeta = {beta}")
    print(f"\tgamma = {gamma}")
    print(f"\tr-value = {r_value_b}")  

    print()

    print("Similarity model:")
    feature_weights, r_value_s = fit_similarity(df)
    print(f"\tfeature weights = {feature_weights}")
    print(f"\tr-value = {r_value_s}")

    print()

    
    best_model = max([
        (r_value_l, "Likelihood Model"),  
        (r_value_b, "Bayesian Model"),
        (r_value_s, "Similarity Model"),
    ])
    print(f"Best Model: {best_model[1]}")

def main():
    g1_df, g2_df = get_groups()

    print("---- GROUP 1 ANALYSIS ----")
    evaluate_models(g1_df)

    print("\n")

    print("---- GROUP 2 ANALYSIS ----")
    evaluate_models(g2_df)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()