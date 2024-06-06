import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    url = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.data'
    column_names =['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(url, sep='\t', names=column_names)
    return df

def preprocess_data(df):
    df = df.drop('timestamp', axis=1)
    df = ordenar_df(df)
    train_data, test_data = train_test_split(df, test_size=0.20)
    return train_data, test_data

def ordenar_df(df):
    df.sort_values(by=['user_id', 'item_id'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Verificar se o user_id e o item_id não começam com 0
    if df['user_id'].min() != 0 or df['item_id'].min() != 0:
        df['user_id'] = df['user_id'] - 1
        df['item_id'] = df['item_id'] - 1
    
    return df
    