import torch
from mia_lib.config import load_config
from mia_lib.data import get_cifar10_dataloaders, create_subset_dataloader
from mia_lib.models import create_model
from mia_lib.trainer import train_model
from mia_lib.attack.shadow_training import train_shadow_models
from mia_lib.attack.dataset_preparation import create_attack_dataset
from mia_lib.attack.train_attack_model import train_attack_model
import os

def main():
    # 1. Load config.
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Get dataset.
    trainset, testset, trainloader, testloader = get_cifar10_dataloaders(config)
    print("Dataset loaded...")

    # 3. Train the target model on a subset of the test set.
    target_model = create_model(config).to(device)

    # subset indices
    import numpy as np
    total_test_indices = np.arange(len(testset))
    target_train_indices = np.random.choice(total_test_indices, config["training"]["train_subset_size"], replace=False)
    target_eval_indices = np.random.choice(np.setdiff1d(total_test_indices, target_train_indices), config["training"]["eval_subset_size"], replace=False)
    subset_tgt_train_loader = create_subset_dataloader(
        testset,
        target_train_indices,
        batch_size=config["dataset"]["train_batch_size"],
        shuffle=True,
        num_workers=config["dataset"]["num_workers"]
    )
    subset_tgt_eval_loader = create_subset_dataloader(
        testset,
        target_eval_indices,
        batch_size=config["dataset"]["eval_batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"]
    )

    os.makedirs(config["paths"]["model_save_dir"], exist_ok=True)
    target_model_path = os.path.join(config["paths"]["model_save_dir"], "target_model.pth")
    best_acc, best_loss = train_model(target_model, subset_tgt_train_loader, subset_tgt_eval_loader, config, device, target_model_path)
    print(f"Best Val Acc: {best_acc:.4f}, Best Val Loss: {best_loss:.4f}")

    # 4. Train shadow models on the trainig set of CIFAR-10
    #    (or any other datset you want ot use as "shadow" data).
    shadow_models = train_shadow_models(config, trainset, device)

    # 5. For each shadow model, create the member / non-member dataset
    #    typically, combined into "attack dataset"
    import pandas as pd
    df_attack_total = []
    for i, (shadow_model, train_idx, eval_idx, test_idx) in enumerate(shadow_models):
        shadow_train_loader = create_subset_dataloader(
            trainset,
            train_idx,
            batch_size=config["dataset"]["train_batch_size"],
            shuffle=False,
            num_workers=config["dataset"]["num_workers"]
        )
        shadow_test_loader = create_subset_dataloader(
            trainset,
            test_idx,
            batch_size=config["dataset"]["eval_batch_size"],
            shuffle=False,
            num_workers=config["dataset"]["num_workers"]
        )

        df_attack = create_attack_dataset(shadow_model, shadow_train_loader, shadow_test_loader, device, output_dim=config["attack"]["output_dim"])
        df_attack_total.append(df_attack)

        # Free GPU memory
        shadow_model.cpu()
        del shadow_model
    
    df_attack_total = pd.concat(df_attack_total, ignore_index=True)

    # 6. Train the final MIA model.
    attack_model = train_attack_model(df_attack_total, config)


if __name__ == "__main__":
    main()