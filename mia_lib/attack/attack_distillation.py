import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score


class DistillationAttack:
    """
    Distillation-based Membership Inference Attack.
    
    This attack trains a student model to mimic the teacher (target) model,
    then uses the difference in predictions to infer membership.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize Distillation Attack.
        
        Args:
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = device
        self.student_model = None
        
    def train_student(self, teacher_model, student_model, distillation_loader,
                     epochs: int = 5, temperature: float = 3.0, 
                     alpha: float = 0.5, lr: float = 0.001):
        """
        Train student model using knowledge distillation.
        
        Args:
            teacher_model: Target model to mimic
            student_model: Student model to train
            distillation_loader: DataLoader for distillation data
            epochs: Number of training epochs
            temperature: Softmax temperature
            alpha: Blending factor for losses
            lr: Learning rate
        """
        self.student_model = distillation_train_student(
            teacher_model, student_model, distillation_loader, self.device,
            epochs, temperature, alpha, lr
        )
        return self.student_model
        
    def infer_membership(self, teacher_model, test_loader,
                        threshold: float = 0.5, 
                        distance_metric: str = "l1") -> Dict[str, Any]:
        """
        Perform membership inference using distillation attack.
        
        Args:
            teacher_model: Target model
            test_loader: DataLoader with samples to classify
            threshold: Distance threshold for membership
            distance_metric: Distance metric to use
            
        Returns:
            Dictionary with attack results
        """
        if self.student_model is None:
            raise ValueError("Student model not trained. Call train_student first.")
            
        df_results = distillation_attack(
            teacher_model, self.student_model, test_loader, self.device,
            threshold, distance_metric
        )
        
        predictions = df_results['membership_prediction'].values
        distances = df_results['distance'].values
        
        results = {
            'predictions': predictions,
            'distances': distances,
            'threshold': threshold,
            'distance_metric': distance_metric
        }
        
        return results

def distillation_train_student(
        teacher_model,
        student_model,
        loader_for_distillation,
        device,
        epochs=5,
        temperature=3.0,
        alpha=0.5,
        lr=0.001,
        weight_decay=0.0,
):
    """
    Train (distill) a student model to mimic the teacher model (target model).

    :param teacher_model: The pretrained target model (teacher) we want to mimic.
    :param student_model: The student model architecture (initially untrained).
    :param loader_for_distillation: DataLoader for unlabeled (or partially labeled) data
                                    used to distill knowledge from the teacker.
    :param device: 'cpu' or 'cuda'
    :param epochs: number of distillation epochs
    :param temperature: softmax temperature for distillation
    :param alpha: blending factor if you also have ground-truth labels
    :param lr: learning rate for student
    :param weight_decay: optional weight decay
    :return The trained student_model (in-place), also return the student model for convenience.
    """
    teacher_model.eval()
    student_model.train()
    student_model.to(device)

    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr, weight_decay=weight_decay)
    # if you have real labels, you can also define a CrossEntropyLoss for the hard labels
    # For demo, we'll assume purely unlabeld data => purely KL distillation.
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_samples = 0

        for images, maybe_labels in tqdm(loader_for_distillation, desc=f"Distillation Epoch {epoch+1}/{epochs}"):
            images = images.to(device)

            with torch.no_grad():
                teacher_outputs = teacher_model(images) # logits from teacher
                # Soft targets
                soft_targets = F.softmax(teacher_outputs / temperature, dim=1)
            
            # Forward pass (stuent)
            student_outputs = student_model(images) # logits from student
            student_soft = F.log_softmax(student_outputs / temperature, dim=1)

            # Distillation loss = KL(student || teacher_soft) * (temperature**2)
            distillation_loss = F.kl_div(
                student_soft,
                soft_targets,
                reduction="batchmean",
            ) * (temperature ** 2)

            # If you have real labels, e.g. in maybe_labels:
            # ce_loss = cross_entropy(student_outputs, maybe_labels) # hard labels
            # total_loss_val = alpha * distillation_loss + (1 - alpha) * ce_loss
            total_loss_val = distillation_loss

            optimizer.zero_grad()
            
            total_loss_val.backward()
            optimizer.step()

            batch_size = images.shape[0]
            total_loss += total_loss_val.item() * batch_size
            total_samples += batch_size
        
        epoch_loss = total_loss / total_samples
        print(f"[Distillation] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    return student_model

def distillation_attack(
        teacher_model,
        student_model,
        test_loader_for_attack,
        device,
        threshold=0.5,
        distance_metric="l1"
):
    """
    Perform membership inference using difference between teacher & student predictions.

    Steps:
        1) For each sample in 'test_loader_for_attack', get teacher & student predictions.
        2) Compute a 'distance' between teacher & student predictions (e.g., L1 or KL).
        3) If distance < threshold, predict'member' else 'non-member'.
    """
    teacher_model.eval()
    student_model.eval()

    results = []
    sample_index = 0
    epsilon = 1e-8
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader_for_attack, desc="Distillation Attack"):
            images = images.to(device)

            teacher_logits = teacher_model(images)
            student_logits = student_model(images)

            teacher_probs = F.softmax(teacher_logits, dim=1).cpu().numpy()
            student_probs = F.softmax(student_logits, dim=1).cpu().numpy()

            batch_size = images.shape[0]
            for i in range(batch_size):
                t_prob = teacher_probs[i]
                s_prob = student_probs[i]

                # Distance computataion
                if distance_metric == "l1":
                    dist = np.sum(np.abs(t_prob - s_prob))
                elif distance_metric == "l2":
                    dist = np.sqrt(np.sum((t_prob - s_prob) ** 2))
                elif distance_metric == "kl":
                    # KL divergence
                    dist = np.sum(t_prob * np.log((t_prob + epsilon) / (s_prob + epsilon)))
                elif distance_metric == "js":
                    # Jensen-Shannon divergence
                    m = 0.5 * (t_prob + s_prob)
                    kl1 = np.sum(t_prob * np.log((t_prob + epsilon) / (m + epsilon)))
                    kl2 = np.sum(s_prob * np.log((s_prob + epsilon) / (m + epsilon)))
                    dist = 0.5 * (kl1 + kl2)
                else:
                    print(f"Unknown distance metric: {distance_metric}, using L1 by default.")
                    dist = np.sum(np.abs(t_prob - s_prob))
                
                # membership logic: smaller dist => more likely 'member'
                membership_pred = 1 if dist < threshold else 0

                results.append({
                    "sample_index": sample_index,
                    "membership_prediction": membership_pred,
                    "distance": dist
                })
                sample_index += 1
    df_attack = pd.DataFrame(results)
    return df_attack
    