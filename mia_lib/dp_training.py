"""
Differential Privacy Training Module

This module provides comprehensive differential privacy training methods including:
- DP-SGD (custom implementation)
- PATE (Private Aggregation of Teacher Ensembles)
- Output Perturbation
- Standard training (for comparison)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json

from .models import create_model
from .utils import compute_accuracy

logger = logging.getLogger(__name__)


class DPTrainer:
    """Differential Privacy Trainer supporting multiple DP methods."""
    
    def __init__(self, num_classes: int, device: str, pretrained: bool = False, input_channels: int = 3):
        self.num_classes = num_classes
        self.device = device
        self.model = create_model(num_classes, pretrained=pretrained, input_channels=input_channels)
        self.model = self.model.to(device)
    
    def train_standard(self, train_loader: DataLoader, test_loader: DataLoader,
                      epochs: int, lr: float, save_path: str) -> Dict[str, Any]:
        """Train model without differential privacy (baseline)."""
        logger.info("Training standard model (no DP)")
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs//3), gamma=0.1)
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'test_acc': [], 'lr': []
        }
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Record metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            current_lr = optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss / len(test_loader))
            history['val_acc'].append(val_acc)
            history['test_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'accuracy': val_acc
                }, save_path)
            
            scheduler.step()
            
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
                       f"LR: {current_lr:.6f}")
        
        logger.info(f"Best accuracy: {best_acc:.2f}%")
        return history
    
    def train_with_dp_sgd_custom(self, train_loader: DataLoader, test_loader: DataLoader,
                                epochs: int, lr: float, epsilon: float, delta: float,
                                max_grad_norm: float, noise_multiplier: float,
                                save_path: str) -> Dict[str, Any]:
        """Train with custom DP-SGD implementation."""
        logger.info(f"Training with DP-SGD (ε={epsilon}, δ={delta})")
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'test_acc': [], 'lr': [],
            'epsilon': []
        }
        
        best_acc = 0.0
        current_epsilon = 0.0
        
        for epoch in range(epochs):
            # Training phase with DP
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                # Add noise to gradients
                for param in self.model.parameters():
                    if param.grad is not None:
                        noise = torch.normal(0, noise_multiplier * max_grad_norm, 
                                           size=param.grad.shape, device=self.device)
                        param.grad += noise
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Update privacy budget (simplified calculation)
            current_epsilon += 2 * noise_multiplier / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Record metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            current_lr = optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss / len(test_loader))
            history['val_acc'].append(val_acc)
            history['test_acc'].append(val_acc)
            history['lr'].append(current_lr)
            history['epsilon'].append(current_epsilon)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'accuracy': val_acc,
                    'epsilon': current_epsilon
                }, save_path)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
                       f"ε: {current_epsilon:.4f}")
        
        logger.info(f"Best accuracy: {best_acc:.2f}%, Final ε: {current_epsilon:.4f}")
        return history
    
    def train_with_pate(self, teacher_loaders: List[DataLoader], 
                       student_loader: DataLoader, test_loader: DataLoader,
                       teacher_epochs: int, student_epochs: int,
                       epsilon: float, delta: float, save_path: str) -> Dict[str, Any]:
        """Train with Private Aggregation of Teacher Ensembles (PATE)."""
        logger.info(f"Training with PATE (ε={epsilon}, δ={delta})")
        
        num_teachers = len(teacher_loaders)
        logger.info(f"Training {num_teachers} teacher models...")
        
        # Train teacher models
        teacher_models = []
        for i, teacher_loader in enumerate(teacher_loaders):
            logger.info(f"Training teacher {i+1}/{num_teachers}")
            
            teacher_model = create_model(self.num_classes, pretrained=False)
            teacher_model = teacher_model.to(self.device)
            
            optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(teacher_epochs):
                teacher_model.train()
                for batch_x, batch_y in teacher_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = teacher_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            teacher_models.append(teacher_model)
        
        logger.info("Training student model with PATE aggregation...")
        
        # Train student model
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'test_acc': [], 'lr': [],
            'epsilon': []
        }
        
        best_acc = 0.0
        privacy_budget_used = 0.0
        
        for epoch in range(student_epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in student_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Get teacher predictions
                teacher_preds = []
                for teacher in teacher_models:
                    teacher.eval()
                    with torch.no_grad():
                        pred = teacher(batch_x)
                        teacher_preds.append(torch.argmax(pred, dim=1))
                
                # Aggregate with noise (simplified PATE)
                aggregated_labels = []
                for i in range(batch_x.size(0)):
                    votes = [pred[i].item() for pred in teacher_preds]
                    # Add Laplace noise to vote counts
                    vote_counts = np.bincount(votes, minlength=self.num_classes).astype(float)
                    noise = np.random.laplace(0, 1.0, self.num_classes)
                    noisy_counts = vote_counts + noise
                    aggregated_labels.append(np.argmax(noisy_counts))
                
                aggregated_labels = torch.tensor(aggregated_labels, device=self.device)
                
                # Train student
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, aggregated_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
                
                # Update privacy budget (simplified)
                privacy_budget_used += 0.01
            
            # Validation phase
            self.model.eval()
            val_acc = compute_accuracy(self.model, test_loader, self.device)
            
            # Record metrics
            train_acc = 100 * train_correct / train_total
            current_lr = optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(train_loss / len(student_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(0.0)  # Not computed for PATE
            history['val_acc'].append(val_acc)
            history['test_acc'].append(val_acc)
            history['lr'].append(current_lr)
            history['epsilon'].append(privacy_budget_used)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'accuracy': val_acc,
                    'epsilon': privacy_budget_used
                }, save_path)
            
            logger.info(f"Epoch {epoch+1}/{student_epochs}: "
                       f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
                       f"ε used: {privacy_budget_used:.4f}")
        
        logger.info(f"Best accuracy: {best_acc:.2f}%, Final ε: {privacy_budget_used:.4f}")
        return history
    
    def train_with_output_perturbation(self, train_loader: DataLoader, test_loader: DataLoader,
                                     epochs: int, lr: float, epsilon: float, delta: float,
                                     sensitivity: float, save_path: str) -> Dict[str, Any]:
        """Train with output perturbation."""
        logger.info(f"Training with Output Perturbation (ε={epsilon}, δ={delta})")
        
        # First train normally
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'test_acc': [], 'lr': [],
            'epsilon': [epsilon] * epochs  # Fixed epsilon for output perturbation
        }
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                
                # Add noise to outputs for privacy
                noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
                noise = torch.normal(0, noise_scale, size=outputs.shape, device=self.device)
                noisy_outputs = outputs + noise
                
                loss = criterion(noisy_outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)  # Use original outputs for accuracy
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase (without noise)
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Record metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            current_lr = optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss / len(test_loader))
            history['val_acc'].append(val_acc)
            history['test_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'accuracy': val_acc,
                    'epsilon': epsilon
                }, save_path)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
                       f"ε: {epsilon}")
        
        logger.info(f"Best accuracy: {best_acc:.2f}%")
        return history


def create_teacher_data_splits(dataset, num_teachers: int) -> List[DataLoader]:
    """Create disjoint data splits for teacher models in PATE."""
    dataset_size = len(dataset)
    teacher_size = dataset_size // num_teachers
    
    indices = torch.randperm(dataset_size).tolist()
    teacher_loaders = []
    
    for i in range(num_teachers):
        start_idx = i * teacher_size
        if i == num_teachers - 1:  # Last teacher gets remaining data
            end_idx = dataset_size
        else:
            end_idx = (i + 1) * teacher_size
        
        teacher_indices = indices[start_idx:end_idx]
        teacher_subset = Subset(dataset, teacher_indices)
        teacher_loader = DataLoader(teacher_subset, batch_size=128, shuffle=True)
        teacher_loaders.append(teacher_loader)
    
    logger.info(f"Created {num_teachers} teacher data splits")
    return teacher_loaders 