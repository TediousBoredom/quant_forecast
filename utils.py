"""
Utilities for Quantized Diffusion Model
量化扩散模型工具函数

This module provides utility functions for:
1. Checkpoint management
2. Visualization
3. Metrics computation
4. Data preprocessing
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import shutil

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_number(num: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    global_step: int,
    epoch: int,
    config: Dict[str, Any],
    output_path: str,
    keep_last_n: int = 5,
):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        global_step: Current training step
        epoch: Current epoch
        config: Training configuration
        output_path: Path to save checkpoint
        keep_last_n: Number of checkpoints to keep
    """
    checkpoint = {
        'global_step': global_step,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, output_path)
    print(f"Saved checkpoint to {output_path}")
    
    # Clean up old checkpoints
    if keep_last_n > 0:
        checkpoint_dir = output_path.parent
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_step_*.pt'))
        if len(checkpoints) > keep_last_n:
            for ckpt in checkpoints[:-keep_last_n]:
                ckpt.unlink()
                print(f"Removed old checkpoint: {ckpt}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        device: Device to load checkpoint to
        
    Returns:
        Dictionary with checkpoint metadata
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    metadata = {
        'global_step': checkpoint.get('global_step', 0),
        'epoch': checkpoint.get('epoch', 0),
        'config': checkpoint.get('config', {}),
    }
    
    print(f"Loaded checkpoint from step {metadata['global_step']}, epoch {metadata['epoch']}")
    
    return metadata


def visualize_latent_grid(
    latents: torch.Tensor,
    output_path: str,
    nrow: int = 4,
    normalize: bool = True,
):
    """
    Visualize latents as a grid of images.
    
    Args:
        latents: Latents to visualize [B, C, F, H, W]
        output_path: Path to save visualization
        nrow: Number of images per row
        normalize: Whether to normalize latents
    """
    B, C, F, H, W = latents.shape
    
    # Take middle frame
    mid_frame = F // 2
    latent_frame = latents[:, :3, mid_frame].cpu().numpy()  # [B, 3, H, W]
    
    if normalize:
        latent_frame = (latent_frame - latent_frame.min()) / (latent_frame.max() - latent_frame.min() + 1e-8)
    
    # Create grid
    ncol = (B + nrow - 1) // nrow
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 3, ncol * 3))
    
    if ncol == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(B):
        row = i // nrow
        col = i % nrow
        
        img = latent_frame[i].transpose(1, 2, 0)  # [H, W, 3]
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Sample {i}')
    
    # Hide empty subplots
    for i in range(B, ncol * nrow):
        row = i // nrow
        col = i % nrow
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {output_path}")


def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    output_path: str,
):
    """
    Plot training curves.
    
    Args:
        metrics_history: Dictionary of metric lists
        output_path: Path to save plot
    """
    num_metrics = len(metrics_history)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, metrics_history.items()):
        ax.plot(values)
        ax.set_xlabel('Step')
        ax.set_ylabel(name)
        ax.set_title(f'{name} over training')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {output_path}")


def compute_codebook_metrics(
    indices: torch.Tensor,
    num_embeddings: int,
) -> Dict[str, float]:
    """
    Compute codebook usage metrics.
    
    Args:
        indices: Codebook indices [B, F, H, W]
        num_embeddings: Total number of embeddings
        
    Returns:
        Dictionary of metrics
    """
    indices_flat = indices.cpu().numpy().flatten()
    
    # Count unique codes
    unique_codes = np.unique(indices_flat)
    num_used = len(unique_codes)
    usage_ratio = num_used / num_embeddings
    
    # Compute perplexity
    counts = np.bincount(indices_flat, minlength=num_embeddings)
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Remove zeros
    perplexity = np.exp(-np.sum(probs * np.log(probs)))
    
    # Compute entropy
    entropy = -np.sum(probs * np.log(probs))
    
    metrics = {
        'codebook_usage': usage_ratio,
        'codebook_perplexity': perplexity,
        'codebook_entropy': entropy,
        'num_used_codes': num_used,
    }
    
    return metrics


def normalize_latents(
    latents: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize latents to zero mean and unit variance.
    
    Args:
        latents: Input latents [B, C, F, H, W]
        mean: Optional pre-computed mean
        std: Optional pre-computed std
        
    Returns:
        normalized_latents: Normalized latents
        mean: Computed mean
        std: Computed std
    """
    if mean is None:
        mean = latents.mean(dim=(0, 2, 3, 4), keepdim=True)
    
    if std is None:
        std = latents.std(dim=(0, 2, 3, 4), keepdim=True)
    
    normalized = (latents - mean) / (std + 1e-8)
    
    return normalized, mean, std


def denormalize_latents(
    latents: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    Denormalize latents.
    
    Args:
        latents: Normalized latents [B, C, F, H, W]
        mean: Mean used for normalization
        std: Std used for normalization
        
    Returns:
        Denormalized latents
    """
    return latents * std + mean


def create_experiment_dir(
    base_dir: str,
    experiment_name: str,
    config: Dict[str, Any],
) -> Path:
    """
    Create experiment directory with config.
    
    Args:
        base_dir: Base output directory
        experiment_name: Name of experiment
        config: Configuration dictionary
        
    Returns:
        Path to experiment directory
    """
    exp_dir = Path(base_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = exp_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create subdirectories
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'samples').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    
    print(f"Created experiment directory: {exp_dir}")
    
    return exp_dir


def log_model_info(model: nn.Module):
    """Print model information."""
    total_params, trainable_params = count_parameters(model)
    
    print("\n" + "="*60)
    print("Model Information")
    print("="*60)
    print(f"Total parameters: {format_number(total_params)} ({total_params:,})")
    print(f"Trainable parameters: {format_number(trainable_params)} ({trainable_params:,})")
    print(f"Non-trainable parameters: {format_number(total_params - trainable_params)}")
    
    # Estimate model size
    param_size = total_params * 4 / (1024 ** 2)  # Assuming float32
    print(f"Estimated model size: {param_size:.2f} MB")
    print("="*60 + "\n")


def get_gpu_memory_info():
    """Get GPU memory information."""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        total = props.total_memory / (1024 ** 3)
        
        info.append(
            f"GPU {i} ({props.name}): "
            f"{allocated:.2f}GB / {total:.2f}GB allocated, "
            f"{reserved:.2f}GB reserved"
        )
    
    return "\n".join(info)


class MetricsTracker:
    """Track and compute running metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value
            self.counts[key] += 1
    
    def get_average(self) -> Dict[str, float]:
        """Get average of all metrics."""
        return {
            key: self.metrics[key] / self.counts[key]
            for key in self.metrics.keys()
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}
    
    def __str__(self) -> str:
        """String representation."""
        avg = self.get_average()
        return ", ".join([f"{k}={v:.4f}" for k, v in avg.items()])


if __name__ == '__main__':
    # Test utilities
    print("Testing utilities...")
    
    # Test parameter counting
    model = nn.Linear(100, 50)
    total, trainable = count_parameters(model)
    print(f"Linear model: {format_number(total)} parameters")
    
    # Test metrics tracker
    tracker = MetricsTracker()
    tracker.update({'loss': 1.0, 'acc': 0.8})
    tracker.update({'loss': 0.8, 'acc': 0.85})
    print(f"Average metrics: {tracker}")
    
    # Test GPU info
    print(f"\nGPU Info:\n{get_gpu_memory_info()}")
    
    print("\n✓ All utility tests passed!")

