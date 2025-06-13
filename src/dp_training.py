"""
Simplified Differential Privacy Training Module

This module provides basic training with differential privacy methods.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import os
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DPTrainer:
    """
    Simplified Differential Privacy Trainer.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize DP Trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to train on
        """
        self.model = model
        self.device = device
        
    def train_with_dp_sgd_custom(self, train_loader: DataLoader,
                                test_loader: DataLoader,
                                epochs: int = 10,
                                lr: float = 0.01,
                                epsilon: float = 1.0,
                                delta: float = 1e-5,
                                clip_norm: float = 1.0,
                                noise_multiplier: float = 1.0,
                                save_path: Optional[str] = None) -> Dict[str, List]:
        """
        Train with simplified DP-SGD implementation.
        """
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        
        history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(tqdm(train_loader, 
                                                           desc=f"Epoch {epoch+1}/{epochs}")):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                
                # Add noise to gradients
                for param in self.model.parameters():
                    if param.grad is not None:
                        noise = torch.normal(0, noise_multiplier * clip_norm, 
                                           size=param.grad.shape, device=param.grad.device)
                        param.grad.add_(noise)
                
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
                train_total += target.size(0)
            
            # Evaluation
            test_acc = self._evaluate_model(test_loader)
            train_acc = 100. * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Train Acc: {train_acc:.2f}%, "
                       f"Test Acc: {test_acc:.2f}%")
        
        if save_path:
            self._save_model_and_history(save_path, history,
                                       {'method': 'dp_sgd_custom', 'epsilon': epsilon, 'delta': delta})
        
        return history
        
    def train_with_output_perturbation(self, train_loader: DataLoader,
                                     test_loader: DataLoader,
                                     epochs: int = 10,
                                     lr: float = 0.01,
                                     epsilon: float = 1.0,
                                     delta: float = 1e-5,
                                     sensitivity: float = 1.0,
                                     save_path: Optional[str] = None) -> Dict[str, List]:
        """
        Train with output perturbation (add noise to model outputs).
        """
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Calculate noise scale for Gaussian mechanism
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
        history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(tqdm(train_loader,
                                                           desc=f"Epoch {epoch+1}/{epochs}")):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                
                # Add noise to outputs during training
                if self.model.training:
                    noise = torch.normal(0, noise_scale, size=output.shape, device=output.device)
                    noisy_output = output + noise
                    loss = criterion(noisy_output, target)
                else:
                    loss = criterion(output, target)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
                train_total += target.size(0)
            
            # Evaluation (without noise)
            test_acc = self._evaluate_model(test_loader)
            train_acc = 100. * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Train Acc: {train_acc:.2f}%, "
                       f"Test Acc: {test_acc:.2f}%")
        
        if save_path:
            self._save_model_and_history(save_path, history,
                                       {'method': 'output_perturbation', 'epsilon': epsilon, 'delta': delta})
        
        return history
        
    def train_standard(self, train_loader: DataLoader,
                      test_loader: DataLoader,
                      epochs: int = 10,
                      lr: float = 0.01,
                      save_path: Optional[str] = None) -> Dict[str, List]:
        """
        Train without differential privacy (baseline).
        """
        return self._train_standard_model(self.model, train_loader, epochs, lr, test_loader, save_path)
        
    def _train_standard_model(self, model: nn.Module, train_loader: DataLoader,
                             epochs: int, lr: float = 0.01,
                             test_loader: Optional[DataLoader] = None,
                             save_path: Optional[str] = None) -> Dict[str, List]:
        """Helper method for standard training."""
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        history = {'train_loss': [], 'train_acc': []}
        if test_loader:
            history['test_acc'] = []
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
                train_total += target.size(0)
            
            train_acc = 100. * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            
            log_msg = f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%"
            
            if test_loader:
                test_acc = self._evaluate_model(test_loader, model)
                history['test_acc'].append(test_acc)
                log_msg += f", Test Acc: {test_acc:.2f}%"
            
            logger.info(log_msg)
        
        if save_path:
            self._save_model_and_history(save_path, history, {'method': 'standard'})
        
        return history
        
    def _evaluate_model(self, test_loader: DataLoader, model: Optional[nn.Module] = None) -> float:
        """Evaluate model accuracy."""
        if model is None:
            model = self.model
            
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return 100. * correct / total
        
    def _save_model_and_history(self, save_path: str, history: Dict, metadata: Dict):
        """Save model and training history."""
        # Save model
        torch.save(self.model.state_dict(), save_path)
        
        # Save history and metadata
        save_dir = Path(save_path).parent
        save_name = Path(save_path).stem
        
        history_path = save_dir / f"{save_name}_history.json"
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = {}
            for key, value in history.items():
                if isinstance(value, list):
                    serializable_history[key] = value
                else:
                    serializable_history[key] = str(value)
            
            result = {
                'history': serializable_history,
                'metadata': metadata
            }
            json.dump(result, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
        logger.info(f"History saved to {history_path}")


def create_teacher_data_splits(dataset, num_teachers: int = 5, 
                              random_seed: int = 42) -> List[DataLoader]:
    """
    Split dataset into multiple teacher datasets for PATE.
    
    Args:
        dataset: Original dataset
        num_teachers: Number of teacher models
        random_seed: Random seed for reproducibility
        
    Returns:
        List of DataLoaders for teacher training
    """
    torch.manual_seed(random_seed)
    
    # Calculate split sizes
    total_size = len(dataset)
    teacher_size = total_size // num_teachers
    sizes = [teacher_size] * num_teachers
    
    # Adjust for remainder
    remainder = total_size - sum(sizes)
    for i in range(remainder):
        sizes[i] += 1
    
    # Split dataset
    teacher_datasets = random_split(dataset, sizes)
    
    # Create data loaders
    teacher_loaders = []
    for teacher_dataset in teacher_datasets:
        loader = DataLoader(teacher_dataset, batch_size=128, shuffle=True, num_workers=2)
        teacher_loaders.append(loader)
    
    return teacher_loaders 