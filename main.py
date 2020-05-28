from automl_pipeline import config, features, data, text, model
import sklearn
from sklearn.model_selection import train_test_split
import autosklearn.classification


print('Loading automl pipeline settings')
settings = config.load_pipeline()

print('Reading data from files')
df = text.get_pt_br_lyrics(data.load_lyrics())

print('Generating features and target')
features, target = features.run_pipeline(df, settings.get('features'))

print('Vanilla RandomForest')
rf = model.run_rf(features, target)

# print('Modelling')
# automl = model.run_automl(settings.get('model'), features, target)