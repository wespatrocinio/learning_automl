from langdetect import detect

import string
import re

def detect_language(col):
    return col.apply(lambda x: detect(x))

def get_pt_br_lyrics(df):
    df['language'] = detect_language(df['lyric'])
    return df[df['language'] == 'pt'][['lyric', 'genre']]
