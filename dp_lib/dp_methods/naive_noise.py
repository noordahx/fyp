# minimal example of Gaussian noise to the final trained model's weights

import torch

def add_noise_to_model(model, sigma=0.01):
    """
    Adds Gaussian noise to each parameter in the model
    as a native approach to final-weight DP.
    """
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.normal(0, sigma, size=param.shape, device=param.device)
            param += noise
    return model