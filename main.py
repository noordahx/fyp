# %%
import torch
import os
import numpy as np
import pandas as pd

from mia_lib.config import load_config
from mia_lib.data import get_cifar10_dataloaders, create_subset_dataloader
from mia_lib.models import create_model
from mia_lib.trainer import train_model
from mia_lib.attack.shadow_training import train_shadow_models
from mia_lib.attack.dataset_preparation import create_attack_dataset
from mia_lib.attack.train_attack_model import train_attack_model

# %% [markdown]
# ### Load config.

# %%
config = load_config("configs/mia_config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ### Get dataset

# %%
trainset, testset, trainloader, testloader = get_cifar10_dataloaders(config)
print("Dataset loaded...")

# %% [markdown]
# ### Train (or load existing) target model (resnet18 in our case) on a subset of the test set.

# %%
target_model = create_model(config).to(device)
os.makedirs(config["paths"]["model_save_dir"], exist_ok=True)

target_model_path = os.path.join(config["paths"]["model_save_dir"], "target_model.pth")

if os.path.exists(target_model_path):
    print(f"Target model checkpoint found at {target_model_path}. Loading...")
    target_model.load_state_dict(torch.load(target_model_path))
else:
    print(f"No target model checkoint found, Training a new one at {target_model_path}")
    # subset indices
    total_test_indices = np.arange(len(testset))
    
    # some MIA research workfloas reserve a "train" portion for shadow models. So, train model on test subset
    target_train_indices = np.random.choice(
        total_test_indices,
        config["training"]["train_subset_size"],
        replace=False
    )

    remaining_after_train = np.setdiff1d(total_test_indices, target_train_indices)
    target_eval_indices = np.random.choice(
        remaining_after_train,
        config["training"]["eval_subset_size"],
        replace=False
    )

    subset_tgt_train_loader = create_subset_dataloader(
        testset,
        target_train_indices,
        batch_size=config["training"]["train_batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )

    subset_tgt_eval_loader = create_subset_dataloader(
        testset,
        target_eval_indices,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"]
    )

    best_acc, best_loss = train_model(
        target_model,
        subset_tgt_train_loader,
        subset_tgt_eval_loader,
        config,
        device,
        target_model_path
    )

    print(f"Target Model => Best Val Acc: {best_acc:.4f}%, Best Val Loss: {best_loss:.4f}")


# %% [markdown]
# ### Train (or load) shadow models on the training set of CIFAR-10.

# %%
shadow_models = train_shadow_models(config, trainset, device)

# %% [markdown]
# ### For each shadow model, create the member/non-member dataset

# %%
df_attack_total = []

for i, (shadow_model, train_idx, eval_idx, test_idx) in enumerate(shadow_models):
    # Rebuild DataLoaders from Membership dataset creation
    shadow_train_loader = create_subset_dataloader(
        trainset,
        train_idx,
        batch_size=config["dataset"]["train_batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"]
    )

    shadow_test_loader = create_subset_dataloader(
        trainset,
        eval_idx,
        batch_size=config["dataset"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"]
    )

    # Create the attack dataset for the shadow model
    df_attack = create_attack_dataset(
        shadow_model,
        shadow_train_loader,
        shadow_test_loader,
        device,
        output_dim=config["attack"]["output_dim"]
    )
    
    df_attack_total.append(df_attack)

    # Free GPU mem
    shadow_model.cpu()
    del shadow_model

df_attack_total = pd.concat(df_attack_total, ignore_index=True)

# %% [markdown]
# ### Train (or load) the final MIA model

# %%
attack_save_dir = config["paths"]["attack_save_dir"]
os.makedirs(attack_save_dir, exist_ok=True)

attack_model_path = os.path.join(attack_save_dir, "attack_model.pth")

if os.path.exists(attack_model_path):
    print(f"Attack model found at {attack_model_path}. Loading...")
else:
    attack_model = train_attack_model(df_attack_total, config)


