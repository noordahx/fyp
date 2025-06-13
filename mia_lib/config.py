# parse config.yaml

import yaml
from easydict import EasyDict

def load_config(config_name, config_path):
    with open(config_path) as infile:
        SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
        CFG = EasyDict(SAVED_CFG[config_name])
    return CFG