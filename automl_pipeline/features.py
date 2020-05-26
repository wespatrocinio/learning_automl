from sklearn import datasets

def get_example_features():
    print('Loading data')
    return datasets.load_breast_cancer(return_X_y=True)