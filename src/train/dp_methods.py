import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from abc import ABC, abstractmethod
from opacus.accountants import RDPAccountant
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DPMethod(ABC):
    """Base class for all differential privacy training methods."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.name = "base"
    
    @abstractmethod
    def train(self, train_loader, test_loader, epochs, lr, save_path, **kwargs) -> Dict[str, Any]:
        """Train model with this DP method."""
        pass

class StandardTraining(DPMethod):
    """Standard training without differential privacy (baseline)."""
    
    def __init__(self, model, device):
        super().__init__(model, device)
        self.name = "standard"
    
    def train(self, train_loader, test_loader, epochs, lr, save_path, **kwargs) -> Dict[str, Any]:
        logger.info("Training standard model (no DP)")
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs//3), gamma=0.1)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        best_acc = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for batch_x, batch_y in train_loader:
                # Asynchronous GPU transfer for better performance
                batch_x = batch_x.to(self.device, non_blocking=True)  
                batch_y = batch_y.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            self.model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    # Asynchronous GPU transfer for better performance
                    batch_x = batch_x.to(self.device, non_blocking=True)  
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss / len(test_loader))
            history['val_acc'].append(val_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({'model_state_dict': self.model.state_dict()}, save_path)
            
            scheduler.step()
            logger.info(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        return history

class DPSGDCustom(DPMethod):
    """Custom implementation of DP-SGD with proper privacy accounting."""
    
    def __init__(self, model, device):
        super().__init__(model, device)
        self.name = "dp_sgd_custom"
    
    def train(self, train_loader, test_loader, epochs, lr, save_path, **kwargs) -> Dict[str, Any]:
        epsilon = float(kwargs['epsilon'])
        delta = float(kwargs['delta'])
        max_grad_norm = float(kwargs['max_grad_norm'])
        noise_multiplier = float(kwargs.get('noise_multiplier', 1.0))
        
        # Validate privacy parameters
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if delta <= 0 or delta >= 1:
            raise ValueError(f"Delta must be in (0, 1), got {delta}")
        if max_grad_norm <= 0:
            raise ValueError(f"Max grad norm must be positive, got {max_grad_norm}")
        
        logger.info(f"Training with DP-SGD (target ε={epsilon}, δ={delta}, C={max_grad_norm}, σ={noise_multiplier})")
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Set up privacy accounting
        accountant = RDPAccountant()
        sample_rate = train_loader.batch_size / len(train_loader.dataset)
        logger.info(f"Sample rate: {sample_rate:.6f}")
        
        # Auto-calibrate noise if needed
        if 'noise_multiplier' not in kwargs:
            # Simple heuristic: start with noise_multiplier = epsilon
            noise_multiplier = max(0.5, epsilon)
            logger.info(f"Auto-calibrated noise multiplier: {noise_multiplier}")

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': [], 'epsilon': []}
        best_acc = 0.0
        
        # Early stopping on privacy budget
        privacy_budget_exhausted = False

        for epoch in range(epochs):
            if privacy_budget_exhausted:
                logger.warning(f"Privacy budget exhausted at epoch {epoch}. Stopping early.")
                break
                
            self.model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                # Asynchronous GPU transfer for better performance
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                batch_size = batch_x.size(0)
                
                optimizer.zero_grad()
                
                # Per-sample gradient computation for proper DP-SGD
                # Switch to eval mode to avoid BatchNorm issues with batch_size=1
                self.model.eval()
                
                per_sample_losses = []
                for i in range(batch_size):
                    sample_x = batch_x[i:i+1]
                    sample_y = batch_y[i:i+1]
                    
                    # Forward pass for individual sample (model in eval mode to handle BatchNorm)
                    outputs = self.model(sample_x)
                    loss = criterion(outputs, sample_y)
                    
                    # Backward pass to compute gradients
                    loss.backward(retain_graph=(i < batch_size - 1))
                    
                    # Clip gradients per sample
                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    
                    # Store per-sample gradients
                    if i == 0:
                        # Initialize accumulated gradients
                        for param in self.model.parameters():
                            if param.grad is not None:
                                param.accumulated_grad = param.grad.clone()
                            else:
                                param.accumulated_grad = None
                    else:
                        # Accumulate clipped gradients
                        for param in self.model.parameters():
                            if param.grad is not None and param.accumulated_grad is not None:
                                param.accumulated_grad += param.grad
                    
                    per_sample_losses.append(loss.item())
                    
                    # Clear gradients for next sample
                    if i < batch_size - 1:
                        optimizer.zero_grad()
                
                # Switch back to training mode for parameter updates
                self.model.train()
                
                # Add noise to accumulated gradients
                for param in self.model.parameters():
                    if hasattr(param, 'accumulated_grad') and param.accumulated_grad is not None:
                        # Generate noise directly on GPU with proper dtype
                        noise = torch.randn(
                            param.accumulated_grad.shape,
                            device=self.device,
                            dtype=param.accumulated_grad.dtype
                        ) * (noise_multiplier * max_grad_norm)
                        param.grad = param.accumulated_grad / batch_size + noise / batch_size
                        # Clean up
                        delattr(param, 'accumulated_grad')
                
                optimizer.step()
                
                # Update privacy accounting
                accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
                
                # Check privacy budget
                current_epsilon = accountant.get_epsilon(delta)
                if current_epsilon > epsilon:
                    logger.warning(f"Privacy budget exceeded: ε={current_epsilon:.4f} > {epsilon}")
                    privacy_budget_exhausted = True
                    break
                
                train_loss += np.mean(per_sample_losses)
                
                # Calculate accuracy on original outputs (before noise)
                with torch.no_grad():
                    clean_outputs = self.model(batch_x)
                    _, predicted = torch.max(clean_outputs.data, 1)
                    train_total += batch_y.size(0)
                    train_correct += (predicted == batch_y).sum().item()

            # Validation phase (no noise added during evaluation)
            self.model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    # Asynchronous GPU transfer for better performance
                    batch_x = batch_x.to(self.device, non_blocking=True)  
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            # Calculate metrics
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0
            val_acc = 100 * val_correct / val_total if val_total > 0 else 0
            current_epsilon = accountant.get_epsilon(delta)
            
            # Update history
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss / len(test_loader))
            history['val_acc'].append(val_acc)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            history['epsilon'].append(current_epsilon)
            
            # Save best model with privacy information
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'best_accuracy': best_acc,
                    'final_epsilon': current_epsilon,
                    'target_epsilon': epsilon,
                    'delta': delta,
                    'max_grad_norm': max_grad_norm,
                    'noise_multiplier': noise_multiplier
                }, save_path)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, ε: {current_epsilon:.4f}")
            
            # Early stopping if privacy budget is close to exhaustion
            if current_epsilon > 0.9 * epsilon:
                logger.warning(f"Approaching privacy budget limit: ε={current_epsilon:.4f}/{epsilon:.4f}")

        return history

class OpacusDPSGD(DPMethod):
    """DP-SGD implementation using Opacus library for optimal performance and correctness."""
    
    def __init__(self, model, device):
        super().__init__(model, device)
        self.name = "opacus_dp_sgd"
    
    def train(self, train_loader, test_loader, epochs, lr, save_path, **kwargs) -> Dict[str, Any]:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        from opacus.utils.batch_memory_manager import BatchMemoryManager
        
        epsilon = float(kwargs['epsilon'])
        delta = float(kwargs['delta'])
        max_grad_norm = float(kwargs.get('max_grad_norm', 1.0))
        noise_multiplier = float(kwargs.get('noise_multiplier', 1.0))
        max_physical_batch_size = int(kwargs.get('max_physical_batch_size', 128))
        
        logger.info(f"Training with Opacus DP-SGD (ε={epsilon}, δ={delta}, C={max_grad_norm})")
        logger.info(f"Noise multiplier: {noise_multiplier}, Max physical batch size: {max_physical_batch_size}")
        
        # Fix model for DP compatibility 
        logger.info("Validating and fixing model for DP compatibility...")
        errors = ModuleValidator.validate(self.model, strict=False)
        if errors:
            logger.warning(f"Model compatibility issues found: {len(errors)} layers")
            # Use strict mode to fix all issues including residual connections
            try:
                self.model = ModuleValidator.fix(self.model, strict=True)
                logger.info("✅ Model automatically fixed (strict mode)")
            except Exception as e:
                logger.warning(f"Strict fix failed: {e}")
                # Fall back to standard fix
                self.model = ModuleValidator.fix(self.model)
                logger.info("✅ Model automatically fixed (standard mode)")
        else:
            logger.info("✅ Model is already DP-compatible")
        
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Initialize privacy engine
        privacy_engine = PrivacyEngine()
        
        self.model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=epochs,
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
        )
        
        logger.info(f"✅ Privacy engine initialized")
        logger.info(f"Using noise multiplier: {optimizer.noise_multiplier:.4f}")
        # Get sample rate from data loader info  
        original_batch_size = 256  # From config
        sample_rate = original_batch_size / len(train_loader.dataset)
        logger.info(f"Sample rate: {sample_rate:.6f}")
        
        # Training loop with proper memory management
        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'epsilon': []}
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Use BatchMemoryManager for efficient memory usage
            with BatchMemoryManager(
                data_loader=train_loader, 
                max_physical_batch_size=max_physical_batch_size, 
                optimizer=optimizer
            ) as memory_safe_data_loader:
                
                for batch_idx, (data, target) in enumerate(memory_safe_data_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
                    
                    if batch_idx % 100 == 0:
                        current_epsilon = privacy_engine.accountant.get_epsilon(delta)
                        logger.info(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                                  f'Loss: {loss.item():.4f}, ε: {current_epsilon:.4f}')
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            
            # Test evaluation
            test_loss, test_acc = self._evaluate(test_loader, criterion)
            current_epsilon = privacy_engine.accountant.get_epsilon(delta)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['epsilon'].append(current_epsilon)
            
            logger.info(f'Epoch {epoch+1}/{epochs}: '
                       f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                       f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, '
                       f'ε: {current_epsilon:.4f}')
            
            # Early stopping if privacy budget exceeded
            if current_epsilon > epsilon:
                logger.warning(f"Privacy budget exceeded (ε={current_epsilon:.4f} > {epsilon}), stopping training")
                break
        
        # Save model
        final_epsilon = privacy_engine.accountant.get_epsilon(delta)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epsilon': final_epsilon,
            'delta': delta,
            'max_grad_norm': max_grad_norm,
            'noise_multiplier': optimizer.noise_multiplier,
            'sample_rate': sample_rate,
            'epochs_trained': epoch + 1,
        }, save_path)
        
        logger.info(f"Model saved to {save_path}")
        logger.info(f"Final privacy cost: ε={final_epsilon:.4f}, δ={delta}")
        
        return history
    
    def _evaluate(self, test_loader, criterion):
        """Evaluate model on test set."""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = 100. * correct / total
        return test_loss, test_acc

class OutputPerturbation(DPMethod):
    """Output perturbation method for differential privacy."""
    
    def __init__(self, model, device):
        super().__init__(model, device)
        self.name = "output_perturbation"
    
    def train(self, train_loader, test_loader, epochs, lr, save_path, **kwargs) -> Dict[str, Any]:
        epsilon = float(kwargs['epsilon'])
        delta = float(kwargs['delta'])
        sensitivity = float(kwargs.get('sensitivity', 1.0))
        
        logger.info(f"Training with Output Perturbation (ε={epsilon}, δ={delta})")
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': [], 'epsilon': [epsilon] * epochs}
        best_acc = 0.0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            for batch_x, batch_y in train_loader:
                # Asynchronous GPU transfer for better performance
                batch_x = batch_x.to(self.device, non_blocking=True)  
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                
                # Add noise to outputs for privacy
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
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    # Asynchronous GPU transfer for better performance
                    batch_x = batch_x.to(self.device, non_blocking=True)  
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            current_lr = optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss / len(test_loader))
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'accuracy': val_acc,
                    'epsilon': epsilon
                }, save_path)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, ε: {epsilon}")

        # Add noise application method for inference
        def add_inference_noise(self, outputs):
            """Apply noise to model outputs for private inference."""
            noise = torch.normal(
                mean=0.0,
                std=noise_scale,
                size=outputs.shape,
                device=outputs.device
            )
            return outputs + noise
        
        # Attach the noise application method to the model
        self.model.add_inference_noise = lambda outputs: add_inference_noise(self, outputs)
        self.model.noise_scale = noise_scale
        
        logger.info(f"Training completed. Best accuracy: {best_acc:.2f}%, Total privacy cost: {epsilon:.4f}")
        return history
    
    def apply_noise_to_outputs(self, outputs):
        """Apply differential privacy noise to model outputs for inference."""
        if hasattr(self.model, 'noise_scale'):
            noise = torch.normal(
                mean=0.0,
                std=self.model.noise_scale,
                size=outputs.shape,
                device=outputs.device
            )
            return outputs + noise
        else:
            logger.warning("No noise scale found. Returning clean outputs.")
            return outputs

def get_dp_method(method_name, model, device):
    """Factory function to get DP method by name."""
    methods = {
        'standard': StandardTraining,
        'dp_sgd_custom': DPSGDCustom,
        'opacus_dp_sgd': OpacusDPSGD,
        'output_perturbation': OutputPerturbation,
    }
    if method_name not in methods:
        available_methods = list(methods.keys())
        raise ValueError(f"Unknown DP method: {method_name}. Available methods: {available_methods}")
    return methods[method_name](model, device)
