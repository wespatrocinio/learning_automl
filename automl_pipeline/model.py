from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load

import autosklearn.classification
import sklearn.model_selection
import sklearn.metrics

def split_dataset(features, target):
    return train_test_split(features, target, random_state=1)

def train_automl(settings, features, target):
    print('Loading sklearn classifier')
    print(settings)
    automl = autosklearn.classification.AutoSklearnClassifier(
        **settings
    )
    print('Fitting automl')
    automl.fit(features, target)
    return automl

def predict_and_metrics(model, features, target):
    print('Predicting')
    predicted_target = model.predict(features)
    print(model.show_models())
    print(model.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(target, predicted_target))

def save(model, path):
    dump(model, path)

def load(path):
    return load(path)

def run_automl(settings, features, target, ensemble_size=0):
    train_features, test_features, train_target, test_target = split_dataset(features, target)
    model = train_automl(settings.get('automl'), train_features, train_target)
    if ensemble_size > 0:
        model.fit_ensemble(train_target, ensemble_size=ensemble_size)
    predict_and_metrics(model, test_features, test_target)
    save(model, settings.get('model_path'))
    return model

def run_rf(features, target, ensemble_size=0):
    train_features, test_features, train_target, test_target = split_dataset(features, target)
    model = RandomForestClassifier()
    model.fit(train_features, train_target)
    predicted_target = model.predict(features)
    print("Accuracy score", sklearn.metrics.accuracy_score(target, predicted_target))
    return model