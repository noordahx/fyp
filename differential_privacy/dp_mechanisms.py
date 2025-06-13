#!/usr/bin/env python3
"""
Comprehensive Differential Privacy Mechanisms Module

This module implements various differential privacy mechanisms to protect machine 
learning models from membership inference attacks. It includes mathematical foundations,
privacy accounting, and practical implementations for deep learning.

Mathematical Foundation:
A randomized algorithm M satisfies (ε, δ)-differential privacy if for all datasets D1 and D2 
differing in at most one element, and for all subsets S of Range(M):

    Pr[M(D1) ∈ S] ≤ exp(ε) × Pr[M(D2) ∈ S] + δ

Where:
- ε (epsilon): Privacy budget - smaller values provide stronger privacy
- δ (delta): Failure probability - probability that privacy guarantee fails
- Global sensitivity Δf = max_{D1,D2} ||f(D1) - f(D2)||₁

Author: FYP Project
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import logging
from typing import Tuple, List, Optional, Dict, Any
from scipy import stats
from abc import ABC, abstractmethod
import warnings

logger = logging.getLogger(__name__)


class DPMechanism(ABC):
    """
    Abstract base class for differential privacy mechanisms.
    
    All DP mechanisms must implement the add_noise method and provide
    privacy parameters (epsilon, delta).
    """
    
    def __init__(self, epsilon: float, delta: float = 0.0):
        """
        Initialize DP mechanism.
        
        Args:
            epsilon (float): Privacy budget (smaller = more private)
            delta (float): Failure probability (default: 0 for pure DP)
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if delta < 0 or delta >= 1:
            raise ValueError("Delta must be in [0, 1)")
            
        self.epsilon = epsilon
        self.delta = delta
        
    @abstractmethod
    def add_noise(self, value: np.ndarray, sensitivity: float) -> np.ndarray:
        """
        Add noise to value according to the DP mechanism.
        
        Args:
            value (np.ndarray): Original value
            sensitivity (float): Global sensitivity of the function
            
        Returns:
            np.ndarray: Noisy value satisfying DP
        """
        pass
        
    def get_privacy_params(self) -> Tuple[float, float]:
        """Return privacy parameters (epsilon, delta)."""
        return self.epsilon, self.delta


class LaplaceMechanism(DPMechanism):
    """
    Laplace Mechanism for Differential Privacy.
    
    Mathematical Foundation:
    For a function f: D → ℝᵈ with global sensitivity Δf, the Laplace mechanism
    is defined as: M(D) = f(D) + Lap(Δf/ε)ᵈ
    
    Where Lap(b) is the Laplace distribution with scale parameter b.
    """
    
    def __init__(self, epsilon: float):
        """
        Initialize Laplace mechanism.
        
        Args:
            epsilon (float): Privacy budget
        """
        super().__init__(epsilon, delta=0.0)  # Pure DP
        
    def add_noise(self, value: np.ndarray, sensitivity: float) -> np.ndarray:
        """
        Add Laplace noise to achieve ε-differential privacy.
        
        Args:
            value (np.ndarray): Original value
            sensitivity (float): Global sensitivity Δf
            
        Returns:
            np.ndarray: Value + Lap(Δf/ε)
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, size=value.shape)
        return value + noise
        
    def get_noise_scale(self, sensitivity: float) -> float:
        """Get the scale parameter for Laplace noise."""
        return sensitivity / self.epsilon


class GaussianMechanism(DPMechanism):
    """
    Gaussian Mechanism for Differential Privacy.
    
    Mathematical Foundation:
    For (ε, δ)-DP with δ > 0, the Gaussian mechanism adds noise from N(0, σ²)
    where σ ≥ Δf × √(2 ln(1.25/δ)) / ε
    
    This provides (ε, δ)-differential privacy.
    """
    
    def __init__(self, epsilon: float, delta: float):
        """
        Initialize Gaussian mechanism.
        
        Args:
            epsilon (float): Privacy budget
            delta (float): Failure probability (must be > 0)
        """
        if delta <= 0:
            raise ValueError("Gaussian mechanism requires delta > 0")
        super().__init__(epsilon, delta)
        
    def add_noise(self, value: np.ndarray, sensitivity: float) -> np.ndarray:
        """
        Add Gaussian noise to achieve (ε, δ)-differential privacy.
        
        Args:
            value (np.ndarray): Original value
            sensitivity (float): Global sensitivity Δf
            
        Returns:
            np.ndarray: Value + N(0, σ²)
        """
        sigma = self.get_noise_scale(sensitivity)
        noise = np.random.normal(0, sigma, size=value.shape)
        return value + noise
        
    def get_noise_scale(self, sensitivity: float) -> float:
        """
        Calculate noise scale σ for Gaussian mechanism.
        
        Formula: σ = Δf × √(2 ln(1.25/δ)) / ε
        """
        if self.delta <= 0:
            raise ValueError("Delta must be positive for Gaussian mechanism")
        return sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon


class ExponentialMechanism(DPMechanism):
    """
    Exponential Mechanism for Differential Privacy.
    
    Mathematical Foundation:
    For a utility function u: D × R → ℝ with sensitivity Δu, the exponential
    mechanism selects output r with probability proportional to:
    exp(ε × u(D, r) / (2 × Δu))
    """
    
    def __init__(self, epsilon: float, utility_function, sensitivity: float, 
                 candidates: List[Any]):
        """
        Initialize exponential mechanism.
        
        Args:
            epsilon (float): Privacy budget
            utility_function: Function mapping (dataset, candidate) → utility
            sensitivity (float): Sensitivity of utility function
            candidates (List): List of candidate outputs
        """
        super().__init__(epsilon, delta=0.0)
        self.utility_function = utility_function
        self.sensitivity = sensitivity
        self.candidates = candidates
        
    def select_candidate(self, dataset: Any) -> Any:
        """
        Select candidate according to exponential mechanism.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Selected candidate
        """
        utilities = [self.utility_function(dataset, candidate) 
                    for candidate in self.candidates]
        
        # Calculate probabilities
        scores = [math.exp(self.epsilon * u / (2 * self.sensitivity)) 
                 for u in utilities]
        total_score = sum(scores)
        probabilities = [s / total_score for s in scores]
        
        # Sample according to probabilities
        selected_idx = np.random.choice(len(self.candidates), p=probabilities)
        return self.candidates[selected_idx]
    
    def add_noise(self, value: np.ndarray, sensitivity: float) -> np.ndarray:
        """Not applicable for exponential mechanism."""
        raise NotImplementedError("Exponential mechanism doesn't add noise directly")


class PrivacyAccountant:
    """
    Privacy Accountant for tracking privacy budget consumption.
    
    Implements composition theorems for differential privacy:
    1. Basic Composition: k mechanisms with (εᵢ, δᵢ) give (Σεᵢ, Σδᵢ)
    2. Advanced Composition: Improved bounds for multiple queries
    """
    
    def __init__(self, total_epsilon: float, total_delta: float):
        """
        Initialize privacy accountant.
        
        Args:
            total_epsilon (float): Total privacy budget
            total_delta (float): Total failure probability
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.queries = []
        
    def spend_privacy_budget(self, epsilon: float, delta: float, 
                           query_description: str = "") -> bool:
        """
        Spend privacy budget for a query.
        
        Args:
            epsilon (float): Privacy cost of query
            delta (float): Delta cost of query
            query_description (str): Description of the query
            
        Returns:
            bool: True if budget is available, False otherwise
        """
        if self.spent_epsilon + epsilon > self.total_epsilon:
            logger.warning(f"Insufficient epsilon budget. Need {epsilon}, "
                          f"have {self.total_epsilon - self.spent_epsilon}")
            return False
            
        if self.spent_delta + delta > self.total_delta:
            logger.warning(f"Insufficient delta budget. Need {delta}, "
                          f"have {self.total_delta - self.spent_delta}")
            return False
            
        self.spent_epsilon += epsilon
        self.spent_delta += delta
        self.queries.append({
            'epsilon': epsilon,
            'delta': delta,
            'description': query_description
        })
        
        logger.info(f"Privacy budget spent: ε={epsilon:.4f}, δ={delta:.8f}. "
                   f"Remaining: ε={self.remaining_epsilon():.4f}, "
                   f"δ={self.remaining_delta():.8f}")
        return True
        
    def remaining_epsilon(self) -> float:
        """Return remaining epsilon budget."""
        return max(0, self.total_epsilon - self.spent_epsilon)
        
    def remaining_delta(self) -> float:
        """Return remaining delta budget."""
        return max(0, self.total_delta - self.spent_delta)
        
    def get_advanced_composition_bounds(self, k: int, epsilon_per_query: float, 
                                      delta_per_query: float, 
                                      delta_prime: float) -> Tuple[float, float]:
        """
        Calculate advanced composition bounds.
        
        For k queries each satisfying (ε, δ)-DP, the composition satisfies
        (ε', δ')-DP where:
        ε' = ε × √(2k × ln(1/δ')) + k × ε × (exp(ε) - 1)
        δ' = k × δ + δ'
        
        Args:
            k (int): Number of queries
            epsilon_per_query (float): Epsilon per query
            delta_per_query (float): Delta per query
            delta_prime (float): Additional delta for composition
            
        Returns:
            Tuple[float, float]: (epsilon_total, delta_total)
        """
        if k <= 0:
            return 0.0, 0.0
            
        # Advanced composition theorem
        epsilon_total = (epsilon_per_query * math.sqrt(2 * k * math.log(1 / delta_prime)) + 
                        k * epsilon_per_query * (math.exp(epsilon_per_query) - 1))
        delta_total = k * delta_per_query + delta_prime
        
        return epsilon_total, delta_total


class DPSGDOptimizer:
    """
    Differentially Private Stochastic Gradient Descent.
    
    Mathematical Foundation:
    DP-SGD modifies standard SGD by:
    1. Clipping gradients to bound sensitivity: ḡᵢ = gᵢ / max(1, ||gᵢ||₂ / C)
    2. Adding Gaussian noise: g̃ᵢ = ḡᵢ + N(0, σ² C² I)
    3. Using noise σ = C × √(2 ln(1.25/δ)) / ε for (ε, δ)-DP
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, clip_norm: float,
                 noise_multiplier: float, sample_rate: float, 
                 privacy_accountant: PrivacyAccountant):
        """
        Initialize DP-SGD optimizer.
        
        Args:
            optimizer: Base PyTorch optimizer
            clip_norm (float): Gradient clipping norm C
            noise_multiplier (float): Noise multiplier σ/C
            sample_rate (float): Sampling rate q = batch_size / dataset_size
            privacy_accountant: Privacy accountant for tracking budget
        """
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.privacy_accountant = privacy_accountant
        self.steps = 0
        
    def step(self, loss: torch.Tensor, parameters: List[torch.Tensor]) -> None:
        """
        Perform one DP-SGD step.
        
        Args:
            loss: Loss tensor
            parameters: Model parameters
        """
        # Compute gradients
        gradients = torch.autograd.grad(loss, parameters, create_graph=False)
        
        # Clip gradients per sample (approximation for batch)
        clipped_gradients = []
        for grad in gradients:
            # Per-sample gradient clipping (simplified)
            grad_norm = torch.norm(grad)
            clip_factor = min(1.0, self.clip_norm / (grad_norm + 1e-8))
            clipped_grad = grad * clip_factor
            clipped_gradients.append(clipped_grad)
            
        # Add noise
        noisy_gradients = []
        for clipped_grad in clipped_gradients:
            noise_scale = self.noise_multiplier * self.clip_norm
            noise = torch.normal(0, noise_scale, size=clipped_grad.shape, 
                               device=clipped_grad.device)
            noisy_grad = clipped_grad + noise
            noisy_gradients.append(noisy_grad)
            
        # Update parameters
        for param, noisy_grad in zip(parameters, noisy_gradients):
            if param.grad is not None:
                param.grad.data.copy_(noisy_grad)
            else:
                param.grad = noisy_grad.clone()
                
        self.optimizer.step()
        self.steps += 1
        
    def get_privacy_spent(self, epochs: int, delta: float) -> Tuple[float, float]:
        """
        Calculate privacy spent using RDP accountant (simplified).
        
        Args:
            epochs (int): Number of training epochs
            delta (float): Target delta
            
        Returns:
            Tuple[float, float]: (epsilon, delta) spent
        """
        # Simplified privacy accounting
        # In practice, use more sophisticated RDP accounting
        steps_per_epoch = 1.0 / self.sample_rate
        total_steps = epochs * steps_per_epoch
        
        # Gaussian mechanism analysis
        sigma = self.noise_multiplier
        epsilon = total_steps * self.sample_rate * (
            self.sample_rate / (2 * sigma**2) + 
            math.sqrt(self.sample_rate * math.log(1/delta)) / sigma
        )
        
        return epsilon, delta


class DifferentiallyPrivateModel:
    """
    Wrapper for training models with differential privacy.
    
    Supports multiple DP training methods:
    1. DP-SGD: Gradient perturbation during training
    2. Output perturbation: Add noise to model outputs
    3. PATE: Private Aggregation of Teacher Ensembles
    """
    
    def __init__(self, model: nn.Module, privacy_accountant: PrivacyAccountant):
        """
        Initialize DP model wrapper.
        
        Args:
            model: PyTorch model
            privacy_accountant: Privacy accountant
        """
        self.model = model
        self.privacy_accountant = privacy_accountant
        self.training_history = []
        
    def train_with_dp_sgd(self, train_loader: DataLoader, criterion: nn.Module,
                         optimizer: torch.optim.Optimizer, epochs: int,
                         clip_norm: float = 1.0, noise_multiplier: float = 1.0,
                         delta: float = 1e-5, device: str = 'cpu') -> Dict[str, List]:
        """
        Train model using DP-SGD.
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Base optimizer
            epochs: Number of training epochs
            clip_norm: Gradient clipping norm
            noise_multiplier: Noise multiplier for DP
            delta: Privacy parameter delta
            device: Training device
            
        Returns:
            Dict containing training history
        """
        logger.info("Starting DP-SGD training...")
        
        # Calculate sample rate
        sample_rate = train_loader.batch_size / len(train_loader.dataset)
        
        # Create DP optimizer
        dp_optimizer = DPSGDOptimizer(
            optimizer, clip_norm, noise_multiplier, sample_rate, self.privacy_accountant
        )
        
        self.model.to(device)
        self.model.train()
        
        history = {'loss': [], 'privacy_spent': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                
                # DP-SGD step
                dp_optimizer.step(loss, list(self.model.parameters()))
                
                epoch_loss += loss.item()
                
            # Track privacy spent
            epsilon_spent, delta_spent = dp_optimizer.get_privacy_spent(epoch + 1, delta)
            
            avg_loss = epoch_loss / len(train_loader)
            history['loss'].append(avg_loss)
            history['privacy_spent'].append((epsilon_spent, delta_spent))
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                       f"Privacy spent: (ε={epsilon_spent:.4f}, δ={delta_spent:.8f})")
                       
            # Check privacy budget
            if epsilon_spent > self.privacy_accountant.total_epsilon:
                logger.warning("Privacy budget exceeded! Stopping training.")
                break
                
        logger.info("DP-SGD training completed")
        self.training_history.append(('DP-SGD', history))
        return history
        
    def add_output_noise(self, outputs: torch.Tensor, mechanism: DPMechanism,
                        sensitivity: float) -> torch.Tensor:
        """
        Add differential privacy noise to model outputs.
        
        Args:
            outputs: Model outputs
            mechanism: DP mechanism to use
            sensitivity: Global sensitivity of the function
            
        Returns:
            Noisy outputs satisfying DP
        """
        if not self.privacy_accountant.spend_privacy_budget(
            mechanism.epsilon, mechanism.delta, "Output perturbation"
        ):
            logger.warning("Insufficient privacy budget for output perturbation")
            return outputs
            
        outputs_np = outputs.detach().cpu().numpy()
        noisy_outputs_np = mechanism.add_noise(outputs_np, sensitivity)
        return torch.tensor(noisy_outputs_np, device=outputs.device, dtype=outputs.dtype)
        
    def predict_with_privacy(self, data_loader: DataLoader, mechanism: DPMechanism,
                           sensitivity: float = 1.0, device: str = 'cpu') -> torch.Tensor:
        """
        Make predictions with differential privacy guarantees.
        
        Args:
            data_loader: Data loader
            mechanism: DP mechanism
            sensitivity: Output sensitivity
            device: Device for computation
            
        Returns:
            Noisy predictions
        """
        self.model.eval()
        self.model.to(device)
        
        all_outputs = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(device)
                outputs = self.model(data)
                noisy_outputs = self.add_output_noise(outputs, mechanism, sensitivity)
                all_outputs.append(noisy_outputs)
                
        return torch.cat(all_outputs, dim=0)


class PATEMechanism:
    """
    Private Aggregation of Teacher Ensembles (PATE).
    
    Mathematical Foundation:
    PATE trains multiple "teacher" models on disjoint data subsets and uses
    the exponential mechanism to aggregate their predictions privately.
    
    For k teachers with predictions, the noise added is calibrated to the
    sensitivity of the voting function.
    """
    
    def __init__(self, teachers: List[nn.Module], epsilon: float, delta: float,
                 privacy_accountant: PrivacyAccountant):
        """
        Initialize PATE mechanism.
        
        Args:
            teachers: List of teacher models
            epsilon: Privacy budget per query
            delta: Failure probability per query
            privacy_accountant: Privacy accountant
        """
        self.teachers = teachers
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_accountant = privacy_accountant
        self.query_count = 0
        
    def aggregate_predictions(self, data: torch.Tensor, 
                            device: str = 'cpu') -> torch.Tensor:
        """
        Aggregate teacher predictions using PATE.
        
        Args:
            data: Input data
            device: Computation device
            
        Returns:
            Aggregated private predictions
        """
        if not self.privacy_accountant.spend_privacy_budget(
            self.epsilon, self.delta, f"PATE query {self.query_count}"
        ):
            raise RuntimeError("Insufficient privacy budget for PATE query")
            
        # Get predictions from all teachers
        teacher_predictions = []
        for teacher in self.teachers:
            teacher.eval()
            teacher.to(device)
            with torch.no_grad():
                pred = teacher(data.to(device))
                teacher_predictions.append(torch.argmax(pred, dim=1))
                
        # Convert to numpy for aggregation
        teacher_votes = torch.stack(teacher_predictions, dim=1).cpu().numpy()
        
        # Count votes for each class
        n_classes = max(teacher_votes.max() + 1, 10)  # Assume at least 10 classes
        aggregated_preds = []
        
        for sample_votes in teacher_votes:
            # Count votes
            vote_counts = np.bincount(sample_votes, minlength=n_classes)
            
            # Add Laplace noise for privacy
            noise_scale = 1.0 / self.epsilon  # Sensitivity = 1 for voting
            noisy_counts = vote_counts + np.random.laplace(0, noise_scale, size=n_classes)
            
            # Select class with highest noisy count
            predicted_class = np.argmax(noisy_counts)
            aggregated_preds.append(predicted_class)
            
        self.query_count += 1
        return torch.tensor(aggregated_preds, device=device)
        
    def train_student_model(self, student_model: nn.Module, 
                          unlabeled_data: DataLoader,
                          optimizer: torch.optim.Optimizer,
                          epochs: int = 10, device: str = 'cpu') -> nn.Module:
        """
        Train student model on teacher aggregated labels.
        
        Args:
            student_model: Student model to train
            unlabeled_data: Unlabeled data for distillation
            optimizer: Optimizer for student training
            epochs: Training epochs
            device: Training device
            
        Returns:
            Trained student model
        """
        logger.info("Training PATE student model...")
        
        student_model.to(device)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            student_model.train()
            total_loss = 0.0
            
            for batch_idx, (data, _) in enumerate(unlabeled_data):
                # Get private labels from teachers
                private_labels = self.aggregate_predictions(data, device)
                
                # Train student
                optimizer.zero_grad()
                outputs = student_model(data.to(device))
                loss = criterion(outputs, private_labels.to(device))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(unlabeled_data)
            logger.info(f"PATE Student Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
        logger.info("PATE student training completed")
        return student_model


class PrivacyUtilityAnalyzer:
    """
    Analyze privacy-utility tradeoffs for DP mechanisms.
    """
    
    @staticmethod
    def calculate_utility_loss(clean_accuracy: float, noisy_accuracy: float) -> float:
        """Calculate utility loss due to privacy protection."""
        return max(0, clean_accuracy - noisy_accuracy)
        
    @staticmethod
    def calculate_privacy_gain(baseline_attack_acc: float, protected_attack_acc: float) -> float:
        """Calculate privacy gain from protection mechanism."""
        return max(0, baseline_attack_acc - protected_attack_acc)
        
    @staticmethod
    def privacy_utility_ratio(utility_loss: float, privacy_gain: float) -> float:
        """Calculate privacy-utility ratio."""
        if privacy_gain == 0:
            return float('inf') if utility_loss > 0 else 0
        return utility_loss / privacy_gain
        
    @staticmethod
    def analyze_epsilon_utility_curve(model, test_loader, epsilons: List[float],
                                    delta: float = 1e-5, device: str = 'cpu') -> Dict:
        """
        Analyze how utility changes with privacy budget.
        
        Args:
            model: Model to analyze
            test_loader: Test data
            epsilons: List of epsilon values to test
            delta: Privacy parameter delta
            device: Computation device
            
        Returns:
            Dictionary with epsilon-utility analysis
        """
        results = {'epsilons': epsilons, 'accuracies': [], 'noise_scales': []}
        
        # Calculate baseline accuracy (no noise)
        model.eval()
        model.to(device)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        baseline_accuracy = correct / total
        results['baseline_accuracy'] = baseline_accuracy
        
        # Test different epsilon values
        for epsilon in epsilons:
            mechanism = GaussianMechanism(epsilon, delta)
            sensitivity = 1.0  # Assume unit sensitivity for outputs
            
            correct_noisy = 0
            total_noisy = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    outputs = model(data)
                    
                    # Add DP noise
                    outputs_np = outputs.cpu().numpy()
                    noisy_outputs_np = mechanism.add_noise(outputs_np, sensitivity)
                    noisy_outputs = torch.tensor(noisy_outputs_np, device=device)
                    
                    _, predicted = torch.max(noisy_outputs.data, 1)
                    total_noisy += target.size(0)
                    correct_noisy += (predicted == target).sum().item()
                    
            noisy_accuracy = correct_noisy / total_noisy
            noise_scale = mechanism.get_noise_scale(sensitivity)
            
            results['accuracies'].append(noisy_accuracy)
            results['noise_scales'].append(noise_scale)
            
        return results


def demonstrate_dp_mechanisms():
    """
    Demonstrate different DP mechanisms with examples.
    """
    print("=== Differential Privacy Mechanisms Demonstration ===\n")
    
    # Example 1: Laplace Mechanism
    print("1. Laplace Mechanism")
    print("   Mathematical: M(D) = f(D) + Lap(Δf/ε)")
    
    original_value = np.array([10.0])
    sensitivity = 1.0
    epsilon = 1.0
    
    laplace_mech = LaplaceMechanism(epsilon)
    noisy_value = laplace_mech.add_noise(original_value, sensitivity)
    
    print(f"   Original value: {original_value[0]:.3f}")
    print(f"   Noisy value (ε={epsilon}): {noisy_value[0]:.3f}")
    print(f"   Noise scale: {laplace_mech.get_noise_scale(sensitivity):.3f}\n")
    
    # Example 2: Gaussian Mechanism
    print("2. Gaussian Mechanism")
    print("   Mathematical: M(D) = f(D) + N(0, σ²) where σ ≥ Δf√(2ln(1.25/δ))/ε")
    
    delta = 1e-5
    gaussian_mech = GaussianMechanism(epsilon, delta)
    noisy_value_gaussian = gaussian_mech.add_noise(original_value, sensitivity)
    
    print(f"   Original value: {original_value[0]:.3f}")
    print(f"   Noisy value (ε={epsilon}, δ={delta}): {noisy_value_gaussian[0]:.3f}")
    print(f"   Noise scale: {gaussian_mech.get_noise_scale(sensitivity):.3f}\n")
    
    # Example 3: Privacy Accounting
    print("3. Privacy Accounting")
    print("   Basic Composition: k queries with (εᵢ,δᵢ) → (Σεᵢ, Σδᵢ)")
    
    accountant = PrivacyAccountant(total_epsilon=5.0, total_delta=1e-4)
    
    # Simulate multiple queries
    for i in range(3):
        success = accountant.spend_privacy_budget(1.0, 1e-5, f"Query {i+1}")
        print(f"   Query {i+1}: Success={success}, "
              f"Remaining ε={accountant.remaining_epsilon():.3f}")
    
    print(f"\n   Total privacy spent: ε={accountant.spent_epsilon:.3f}, "
          f"δ={accountant.spent_delta:.8f}")
    
    print("\n=== End of Demonstration ===")


if __name__ == "__main__":
    demonstrate_dp_mechanisms()