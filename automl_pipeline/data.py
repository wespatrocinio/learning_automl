import os
import pandas as pd

GENRES = ['bossa_nova', 'funk', 'gospel', 'sertanejo']

DATA_DIR = 'data/'

def read_file(path):
    return pd.read_csv(path, encoding='utf-8')

def load_lyrics():
    dataframes = []
    for genre in GENRES:
        print(f'reading {genre} file')
        df = read_file(os.path.join(DATA_DIR, f'{genre}.csv'))
        df['genre'] = genre
        dataframes.append(df)
    return pd.concat(dataframes)
    
