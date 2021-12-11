import pandas as pd

DATA_FILENAME = "/Users/juliawagner/MIT_Material/9.660/Project/9.660-final-project/data/data.csv"

def get_sets_to_ratings(is_pop=True):
    df = pd.read_csv(DATA_FILENAME)
    df = df.drop(df.columns[:-70], axis = 1)
    if is_pop:
        df = df.drop(df.columns[-35:], axis=1)
    else:
        df = df.drop(df.columns[:-35], axis = 1)
    sets_to_ratings = {}
    for i,column in enumerate(df.columns):
        songs = column[column.find(':') + 2:].split(', ')
        sets_to_ratings[frozenset(songs)] = df.iloc[:,i].values.tolist()
    return sets_to_ratings