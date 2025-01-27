# DP training code

import torch
import os
from dp_lib.dp_methods.dp_sgd import train_with_dp_sgd
from dp_lib.dp_methods.pate import train_with_pate
from dp_lib.dp_methods.naive_noise import add_noise_to_model

def train_model_with_dp(model, train_loader, val_loader, config, device):
    """
    Dpeending on config['dp']['method'], train the model with the corresponding method.
    """
    dp_method = config["dp"]["method"].lower()
    os.makedirs(config["paths"]["model_save_dir"], exist_ok=True)
