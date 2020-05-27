from os import pipe
from sklearn import datasets
from nltk.tokenize import RegexpTokenizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import nltk

# https://queirozf.com/entries/scikit-learn-pipelines-custom-pipelines-and-pandas-integration
class FeatureTransformer():
    def __init__(self, func):
        self.func = func

    def transform(self, input_df, **transform_params):
        return self.func(input_df)

    def fit(self, X, y=None, **fit_params):
        return self

def get_example_features():
    print('Loading data')
    return datasets.load_breast_cancer(return_X_y=True)

def tokenizer(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    stemmer = nltk.stem.RSLPStemmer()
    return [stemmer.stem(t) for t in tokens]

def get_tokens(input_df):
    input_df['tokens'] = input_df['lyric'].map(lambda x: tokenizer(x))
    return input_df

def get_num_tokens(input_df):
    input_df['num_tokens'] = input_df['lyric'].map(lambda x: len(x))
    return input_df

def get_num_distinct_tokens(input_df):
    input_df['num_distinct_tokens'] = input_df['tokens'].map(lambda x: len(set(x)))
    return input_df

def get_tfidf(input_df):
    vectorizer = TfidfVectorizer(
        encoding='utf-8',
        decode_error='replace',
        strip_accents='unicode',
        analyzer='word',
        binary=False,
        stop_words=nltk.corpus.stopwords.words("portuguese"),
        tokenizer=tokenizer
    )
    tfidf = vectorizer.fit_transform(input_df['lyric'])
    df = pd.DataFrame(tfidf.toarray(), columns=vectorizer.get_feature_names())
    return pd.concat([input_df, df], axis=1, join_axes=[input_df.index])

def convert_target(category, category_map):
    return category_map.get(category)

def run_pipeline(input_df, settings):
    pipe = Pipeline([
        ('tokens', FeatureTransformer(get_tokens)),
        ('num_tokens', FeatureTransformer(get_num_tokens)),
        ('num_distinct_tokens', FeatureTransformer(get_num_distinct_tokens)),
        ('tfidf', FeatureTransformer(get_tfidf))
    ])
    df = pipe.fit_transform(input_df)
    target = df['genre'].apply(lambda x: convert_target(x, settings.get('target_categories_map')))
    features = df.drop(columns=['lyric', 'genre', 'tokens'])
    return features, target
