from pathlib import Path

import yaml


def extract_configs(filename: Path) -> dict:
    """Utility function serves to read YAML files and load them into python dict.
    """
    with open(filename) as file:
        content = yaml.load(file, Loader=yaml.FullLoader)
    return content
