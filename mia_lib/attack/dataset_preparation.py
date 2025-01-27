# logic to create the member vs. non-member dataset from shadow models

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

def make_member_nonmember_dataset(model, train_loader, test_loader, device):
    """
    Returns two numpy arrays, one for members, one for non-members,
    contains the top-10 probabilities, and a label "is_member".
    """
    model.eval()
    member_probs = []
    non_member_probs = []

    with torch.no_grad():
        # Member
        for images, _ in tqdm(train_loader, desc="Member set"):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            top_p, _ = probs.topk(10, dim=1)
            member_probs.append(top_p.cpu().numpy())

        # Non-member
        for images, _ in tqdm(test_loader, desc="Non-member set"):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            top_p, _ = probs.topk(10, dim=1)
            non_member_probs.append(top_p.cpu().numpy())

        member_probs = np.concatenate(member_probs, axis=0)
        non_member_probs = np.concatenate(non_member_probs, axis=0)

        return member_probs, non_member_probs

def create_attack_dataset(shadow_model, train_loader, test_loader, device, output_dim=10):
    """
    Creates a DataFrame with columns = top_{index}_prob + [is_member].
    """
    columns = [f"top_{i}_prob" for i in range(output_dim)]

    member_probs, non_member_probs = make_member_nonmember_dataset(
        shadow_model, train_loader, test_loader, device
    )

    df_member = pd.DataFrame(member_probs, columns=columns)
    df_member["is_member"] = 1

    df_non_member = pd.DataFrame(non_member_probs, columns=columns)
    df_non_member["is_member"] = 0

    df = pd.concat([df_member, df_non_member], ignore_index=True)
    return df

