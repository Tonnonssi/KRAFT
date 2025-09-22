import pickle

def get_df(path):
    with open(path, 'rb') as f:
        df = pickle.load(f)
    return df