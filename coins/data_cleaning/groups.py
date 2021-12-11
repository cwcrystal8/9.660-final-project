import pandas as pd
import plotly.express as px

from ..sequence_info.sequences import group_2_coins
from ..utils import google_form_question_to_coin_sequence

DATA_FILENAME = "C:/Users/Crystal Wang/Downloads/9.660/9.660-final-project/data/data.csv"

def get_df():
    df = pd.read_csv(DATA_FILENAME)
    df = df.drop("Timestamp", axis = 1)

    # Get rid of the music data
    df = df.drop(df.columns[-70:], axis = 1)

    # Clean ratings
    df = df.replace("1 (least representative)", 1)
    df = df.replace("7 (most representative)", 7)

    # Set index
    df = df.set_index("Name")
    return df


def split_groups(df):
    group1_df = df[df["Who asked you to do this survey?"] == "Crystal (Group 1)"]
    group1_df = group1_df.drop(group1_df.columns[-37:], axis = 1)
    group1_df = group1_df.drop("Who asked you to do this survey?", axis = 1)

    group2_df = df[df["Who asked you to do this survey?"] == "Julia (Group 2)"]
    group2_df = group2_df.drop(group2_df.columns[-73: -36], axis = 1)
    group2_df = group2_df.drop("Who asked you to do this survey?", axis = 1)
    return group1_df, group2_df

def get_control_data(g_df):
    g_control_df = g_df.drop(g_df.columns[5:], axis = 1)
    g_control_df = g_control_df.astype(int)
    g_control_df.columns = [1,2,3,4,5]
    return g_control_df

def plot_line_data(df):
    df = df.stack().reset_index()
    df = df.rename(columns = {"level_1": "x", 0: "y"})
    px.line(df, x = "x", y = "y", color = "Name")

def significance_t_test(df1, df2):
    df1_means = df1.mean(axis = 0)
    df2_means = df2.mean(axis = 0)
    
    df1_vars = df1.std(axis = 0, ddof = 0) ** 2
    df2_vars = df2.std(axis = 0, ddof = 0) ** 2
    
    df1_vars /= len(df1.index)
    df2_vars /= len(df2.index)
    
    numerator = df1_means - df2_means
    denominator = (df1_vars + df2_vars) ** 0.5

    return (numerator / denominator).abs() > 1.96
    
def verify_control_significance(g1_df, g2_df, plot = False):
    g1_control_df = get_control_data(g1_df)
    g2_control_df = get_control_data(g2_df)

    if plot:
        plot_line_data(g1_control_df)
        plot_line_data(g2_control_df)

    # See if there is a significant difference for each sequence
    is_significant = significance_t_test(g1_control_df, g2_control_df)
    
    if is_significant.any():
        raise Exception("The samples are significantly different in control responses!")
    else:
        print("No significant difference between the control responses.\n")

    return


def remove_control_data(g_df):
    return g_df.drop(g_df.columns[:5], axis = 1).astype(int)


def sort_columns(df):
    return df.reindex(sorted(df.columns), axis=1)


def clean_group_1(g1_df):
    g1_df.columns = [google_form_question_to_coin_sequence(column) for column in g1_df.columns]
    g1_df = sort_columns(g1_df)
    return g1_df


def clean_group_2(g2_df):
    g2_df.columns = [
        google_form_question_to_coin_sequence(f"{column} [Coin {coin}: ]") 
        for column, coin in zip(g2_df.columns, group_2_coins)
    ]
    g2_df = sort_columns(g2_df)
    return g2_df


def test_experiment_significance(g1_df, g2_df):
    print("---- GROUP 1 vs. GROUP 2 EXPERIMENT ----")
    is_significant = significance_t_test(g1_df, g2_df)

    if is_significant.any():
        significant_cols = list((is_significant.loc[is_significant]).index)
        print(f"There is a significant difference between group 1 and group 2: {significant_cols}\n")

    else:
        print("There is NO significant difference between group 1 and group 2\n")


def get_groups():
    df = get_df()
    g1_df, g2_df = split_groups(df)
    
    verify_control_significance(g1_df, g2_df)
    
    g1_df = remove_control_data(g1_df)
    g2_df = remove_control_data(g2_df)
    
    g1_df = clean_group_1(g1_df)
    g2_df = clean_group_2(g2_df)

    test_experiment_significance(g1_df, g2_df)
    
    return g1_df, g2_df