"""
Differential Privacy Training Orchestrator

This module orchestrates the training process using the specified differential privacy methods.
"""


import logging
from pathlib import Path
import torch

from src.models import create_model
from src.data import get_dataset
from src.train.dp_methods import get_dp_method
from src.visualization import plot_training_metrics

logger = logging.getLogger(__name__)

def run_training(config, device):
    """
    Main function to run the training pipeline based on the provided config.
    
    Args:
        config: Configuration object containing all the parameters for training.
        device: Device to run the training on (e.g., 'cuda' or 'cpu').
    """
    logger.info(f"Loading dataset: {config.data.dataset}")
    train_dataset, test_dataset = get_dataset(config.data.dataset, './data')
    logger.info(f"Dataset loaded successfully: Train={len(train_dataset)}, Test={len(test_dataset)}")
    
    # Optimized DataLoaders for GPU training
    is_cuda = device.type == 'cuda'
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.data.train_batch_size, 
        shuffle=True, 
        num_workers=max(config.data.num_workers, 4 if is_cuda else 0),
        pin_memory=is_cuda,
        drop_last=True  # Consistent batch sizes for DP training
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config.data.eval_batch_size, 
        shuffle=False, 
        num_workers=max(config.data.num_workers, 4 if is_cuda else 0),
        pin_memory=is_cuda
    )

    logger.info(f"Creating model: {config.model.architecture}")
    model = create_model(config.model)
    logger.info(f"Model created successfully, moving to device: {device}")
    model = model.to(device)

    logger.info(f"Initializing DP method: {config.defense.method}")
    dp_method = get_dp_method(config.defense.method, model, device)
    logger.info(f"DP method initialized: {dp_method.name}")
    
    output_dir = Path(config.output.save_dir)
    model_name = f"{config.data.dataset}_{config.defense.method}_eps{config.defense.epsilon}.pt"
    save_path = output_dir / model_name

    logger.info(f"Starting training with method: {dp_method.name}")
    history = dp_method.train(
        train_loader,
        test_loader,
        epochs=config.training.epochs,
        lr=config.training.learning_rate,
        save_path=save_path,
        epsilon=config.defense.epsilon,
        delta=config.defense.delta,
        max_grad_norm=config.defense.max_grad_norm,
        noise_multiplier=config.defense.noise_multiplier
    )

    if config.output.visualizations:
        plot_path = output_dir / f"{Path(model_name).stem}_training.png"
        plot_training_metrics(history, plot_path, method=dp_method.name, epsilon=config.defense.epsilon)
        logger.info(f"Training visualization saved to {plot_path}")

    logger.info("Training finished.")