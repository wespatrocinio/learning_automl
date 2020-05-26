import autosklearn.classification
import sklearn.model_selection
import sklearn.metrics

def generic(features, pipeline_settings, persistence):
    X, y = features.get_example_features()
    print('Train/test splitting')
    X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(X, y, random_state=1)

    print('Loading sklearn classifier')
    automl = autosklearn.classification.AutoSklearnClassifier(
        **pipeline_settings.get('pipeline')
    )

    print('Fitting automl')
    print(pipeline_settings)
    automl.fit(X_train, y_train)

    #This call to fit_ensemble uses all models trained in the previous call
    #to fit to build an ensemble which can be used with automl.predict()
    # automl.fit_ensemble(y_train, ensemble_size=50)

    print('Predicting')
    y_hat = automl.predict(X_test)

    print(automl.show_models())
    print(automl.sprint_statistics())
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

    persistence.save_model(automl, 'model/automl.joblib')