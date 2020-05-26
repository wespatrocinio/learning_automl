import yaml

def load(path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_pipeline():
    return load('pipeline_config.yml')