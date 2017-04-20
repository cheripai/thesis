import yaml


def get_config():
    with open("config.yaml") as f:
        config = yaml.load(f)
    return config
