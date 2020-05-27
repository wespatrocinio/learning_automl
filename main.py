from automl_pipeline import config, features, data, persistence, text
import sklearn
from sklearn.model_selection import train_test_split
import autosklearn.classification


print('Loading automl pipeline settings')
model_settings = config.load_pipeline()

print('Reading data from files')
df = text.get_pt_br_lyrics(data.load_lyrics())

print('Generating features and target')
X, y = features.run_pipeline(df, model_settings.get('features'))

print('Train/test splitting')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print('Loading sklearn classifier')
automl = autosklearn.classification.AutoSklearnClassifier(
    **model_settings.get('pipeline')
)

print('Fitting automl')
print(model_settings)
automl.fit(X_train, y_train)

#This call to fit_ensemble uses all models trained in the previous call
#   to fit to build an ensemble which can be used with automl.predict()
automl.fit_ensemble(y_train, ensemble_size=5)

print('Predicting')
y_hat = automl.predict(X_test)

print(automl.show_models())
print(automl.sprint_statistics())
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

persistence.save_model(automl, 'model/automl.joblib')