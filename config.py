"""
Configuration file for Quantized Diffusion Model training
量化扩散模型训练配置文件

This file contains default configurations for training the quantized diffusion model.
You can modify these settings or create custom config files.
"""

import json
from pathlib import Path
from typing import Dict, Any


def get_small_config() -> Dict[str, Any]:
    """Small model configuration for quick testing."""
    return {
        'model': {
            'latent_channels': 16,
            'num_embeddings': 4096,  # Smaller codebook
            'embedding_dim': 256,
            'hidden_dim': 512,
            'num_layers': 6,
            'num_heads': 8,
            'dropout': 0.1,
            'max_frames': 32,
        },
        'diffusion': {
            'num_train_timesteps': 1000,
            'min_step': 20,
            'max_step': 980,
            'beta_schedule': 'linear',
        },
        'training': {
            'num_epochs': 50,
            'num_samples': 5000,
            'batch_size': 8,
            'learning_rate': 2e-4,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'gradient_accumulation_steps': 1,
            'warmup_steps': 500,
            'quantization_weight': 0.1,
            'num_workers': 4,
        },
        'data': {
            'num_frames': 8,
            'latent_height': 32,
            'latent_width': 32,
        },
        'logging': {
            'use_wandb': False,
            'project_name': 'quantized-diffusion-dmd',
            'run_name': 'small_model_test',
            'log_interval': 10,
            'save_interval': 500,
            'max_checkpoints': 3,
        },
        'output_dir': './outputs/quantized_dmd_small',
    }


def get_medium_config() -> Dict[str, Any]:
    """Medium model configuration for standard training."""
    return {
        'model': {
            'latent_channels': 16,
            'num_embeddings': 8192,
            'embedding_dim': 512,
            'hidden_dim': 1024,
            'num_layers': 12,
            'num_heads': 16,
            'dropout': 0.1,
            'max_frames': 64,
        },
        'diffusion': {
            'num_train_timesteps': 1000,
            'min_step': 20,
            'max_step': 980,
            'beta_schedule': 'linear',
        },
        'training': {
            'num_epochs': 100,
            'num_samples': 50000,
            'batch_size': 4,
            'learning_rate': 1e-4,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'gradient_accumulation_steps': 2,
            'warmup_steps': 1000,
            'quantization_weight': 0.1,
            'num_workers': 8,
        },
        'data': {
            'num_frames': 16,
            'latent_height': 32,
            'latent_width': 32,
        },
        'logging': {
            'use_wandb': True,
            'project_name': 'quantized-diffusion-dmd',
            'run_name': 'medium_model_v1',
            'log_interval': 50,
            'save_interval': 1000,
            'max_checkpoints': 5,
        },
        'output_dir': './outputs/quantized_dmd_medium',
    }


def get_large_config() -> Dict[str, Any]:
    """Large model configuration for high-quality generation."""
    return {
        'model': {
            'latent_channels': 16,
            'num_embeddings': 16384,  # Larger codebook
            'embedding_dim': 768,
            'hidden_dim': 2048,
            'num_layers': 24,
            'num_heads': 32,
            'dropout': 0.1,
            'max_frames': 128,
        },
        'diffusion': {
            'num_train_timesteps': 1000,
            'min_step': 20,
            'max_step': 980,
            'beta_schedule': 'cosine',  # Cosine schedule for better quality
        },
        'training': {
            'num_epochs': 200,
            'num_samples': 100000,
            'batch_size': 2,
            'learning_rate': 5e-5,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'gradient_accumulation_steps': 4,
            'warmup_steps': 2000,
            'quantization_weight': 0.05,  # Lower weight for better reconstruction
            'num_workers': 8,
        },
        'data': {
            'num_frames': 32,
            'latent_height': 64,
            'latent_width': 64,
        },
        'logging': {
            'use_wandb': True,
            'project_name': 'quantized-diffusion-dmd',
            'run_name': 'large_model_v1',
            'log_interval': 100,
            'save_interval': 2000,
            'max_checkpoints': 10,
        },
        'output_dir': './outputs/quantized_dmd_large',
    }


def get_dmd_optimized_config() -> Dict[str, Any]:
    """Configuration optimized for DMD training."""
    return {
        'model': {
            'latent_channels': 16,
            'num_embeddings': 8192,
            'embedding_dim': 512,
            'hidden_dim': 1024,
            'num_layers': 12,
            'num_heads': 16,
            'dropout': 0.0,  # No dropout for DMD
            'max_frames': 64,
        },
        'diffusion': {
            'num_train_timesteps': 1000,
            'min_step': 50,  # Higher min_step for DMD
            'max_step': 950,  # Lower max_step for DMD
            'beta_schedule': 'linear',
        },
        'training': {
            'num_epochs': 100,
            'num_samples': 50000,
            'batch_size': 4,
            'learning_rate': 2e-4,  # Higher LR for DMD
            'beta1': 0.0,  # DMD often uses 0.0
            'beta2': 0.999,
            'weight_decay': 0.0,  # No weight decay for DMD
            'max_grad_norm': 1.0,
            'gradient_accumulation_steps': 2,
            'warmup_steps': 500,
            'quantization_weight': 0.1,
            'num_workers': 8,
        },
        'data': {
            'num_frames': 16,
            'latent_height': 32,
            'latent_width': 32,
        },
        'logging': {
            'use_wandb': True,
            'project_name': 'quantized-diffusion-dmd',
            'run_name': 'dmd_optimized_v1',
            'log_interval': 50,
            'save_interval': 1000,
            'max_checkpoints': 5,
        },
        'output_dir': './outputs/quantized_dmd_optimized',
    }


def save_config(config: Dict[str, Any], output_path: str):
    """Save configuration to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved configuration to {output_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded configuration from {config_path}")
    return config


if __name__ == '__main__':
    # Generate and save all default configurations
    configs = {
        'small': get_small_config(),
        'medium': get_medium_config(),
        'large': get_large_config(),
        'dmd_optimized': get_dmd_optimized_config(),
    }
    
    output_dir = Path('./configs')
    output_dir.mkdir(exist_ok=True)
    
    for name, config in configs.items():
        output_path = output_dir / f'config_{name}.json'
        save_config(config, output_path)
    
    print(f"\nGenerated {len(configs)} configuration files in {output_dir}")

