from automl_pipeline import features, text

import os
import pandas as pd

GENRES = ['bossa_nova', 'funk', 'gospel', 'sertanejo']

RAW_DATA_DIR = 'data/raw/'
FEATURES_PATH = 'data/features/features.csv'
TARGET_PATH = 'data/features/target.csv'

def read_file(path):
    return pd.read_csv(path, encoding='utf-8')

def read_raw_data():
    dataframes = []
    for genre in GENRES:
        print(f'reading {genre} file')
        df = read_file(os.path.join(RAW_DATA_DIR, f'{genre}.csv'))
        df['genre'] = genre
        dataframes.append(df)
    return pd.concat(dataframes)
    
def load_features(settings):
    if os.path.isfile(FEATURES_PATH) and os.path.isfile(TARGET_PATH):
        feat = pd.read_csv(FEATURES_PATH, index_col='index')
        target = pd.read_csv(TARGET_PATH, index_col='index')
    else:
        df = text.get_pt_br_lyrics(read_raw_data())
        feat, target = features.run_pipeline(df, settings)
    return feat, target
