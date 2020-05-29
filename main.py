from automl_pipeline import config, features, data, text, model

if __name__ == '__main__':
    print('Loading automl pipeline settings')
    settings = config.load_pipeline()

    print('Generating features and target')
    features, target = data.load_features(settings.get('features'))

    print(features.shape, target.shape)

    # print('Vanilla RandomForest')
    # rf = model.run_rf(features, target)

    # print('Vanilla DecisionTree')
    # rf = model.run_dt(features, target)

    print('Modelling')
    automl = model.run_automl(settings.get('model'), features, target)