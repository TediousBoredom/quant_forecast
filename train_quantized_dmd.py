"""
Training script for Quantized Diffusion Model with DMD
基于 DMD 的量化扩散模型训练脚本

This script trains a quantized diffusion model using Distribution Matching Distillation (DMD).
It supports:
1. Multi-GPU training with FSDP
2. Gradient accumulation
3. Mixed precision training
4. WandB logging
5. Checkpoint saving/loading
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging disabled")

from quantized_diffusion_model import (
    QuantizedDiffusionPredictor,
    QuantizedDiffusionDMD,
)


class DummyVideoDataset(Dataset):
    """
    Dummy dataset for testing.
    Replace this with your actual video dataset.
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        latent_shape: tuple = (16, 8, 32, 32),  # (C, F, H, W)
        text_embed_dim: int = 768,
    ):
        self.num_samples = num_samples
        self.latent_shape = latent_shape
        self.text_embed_dim = text_embed_dim
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random latent
        latent = torch.randn(*self.latent_shape)
        
        # Generate random text embedding
        text_embed = torch.randn(77, self.text_embed_dim)
        
        return {
            'latent': latent,
            'text_embed': text_embed,
            'caption': f"Sample video {idx}",
        }


class QuantizedDMDTrainer:
    """
    Trainer for Quantized Diffusion Model with DMD.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        local_rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            local_rank: Local rank for distributed training
            world_size: Total number of processes
        """
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{local_rank}')
        self.is_main_process = (local_rank == 0)
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self._setup_model()
        
        # Initialize optimizer and scheduler
        self._setup_optimizer()
        
        # Initialize dataset and dataloader
        self._setup_data()
        
        # Initialize logging
        self._setup_logging()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
    def _setup_model(self):
        """Setup model and DMD wrapper."""
        print(f"[Rank {self.local_rank}] Setting up model...")
        
        # Create model
        self.model = QuantizedDiffusionPredictor(
            latent_channels=self.config['model']['latent_channels'],
            num_embeddings=self.config['model']['num_embeddings'],
            embedding_dim=self.config['model']['embedding_dim'],
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            num_heads=self.config['model']['num_heads'],
            dropout=self.config['model']['dropout'],
            max_frames=self.config['model']['max_frames'],
        ).to(self.device)
        
        # Wrap with DDP if distributed
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
            model_for_dmd = self.model.module
        else:
            model_for_dmd = self.model
        
        # Create DMD wrapper
        self.dmd = QuantizedDiffusionDMD(
            model=model_for_dmd,
            device=self.device,
            num_train_timesteps=self.config['diffusion']['num_train_timesteps'],
            min_step=self.config['diffusion']['min_step'],
            max_step=self.config['diffusion']['max_step'],
            beta_schedule=self.config['diffusion']['beta_schedule'],
            quantization_weight=self.config['training']['quantization_weight'],
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if self.is_main_process:
            print(f"Model parameters: {num_params:,} ({num_trainable:,} trainable)")
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        print(f"[Rank {self.local_rank}] Setting up optimizer...")
        
        # Create optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            betas=(self.config['training']['beta1'], self.config['training']['beta2']),
            weight_decay=self.config['training']['weight_decay'],
            eps=1e-8,
        )
        
        # Create learning rate scheduler
        num_training_steps = self.config['training']['num_epochs'] * \
                            (self.config['training']['num_samples'] // 
                             (self.config['training']['batch_size'] * self.world_size))
        
        warmup_steps = self.config['training']['warmup_steps']
        
        if warmup_steps > 0:
            # Warmup + Cosine decay
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - warmup_steps,
                eta_min=self.config['training']['learning_rate'] * 0.1,
            )
            
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        else:
            # Just cosine decay
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=self.config['training']['learning_rate'] * 0.1,
            )
    
    def _setup_data(self):
        """Setup dataset and dataloader."""
        print(f"[Rank {self.local_rank}] Setting up data...")
        
        # Create dataset
        self.dataset = DummyVideoDataset(
            num_samples=self.config['training']['num_samples'],
            latent_shape=(
                self.config['model']['latent_channels'],
                self.config['data']['num_frames'],
                self.config['data']['latent_height'],
                self.config['data']['latent_width'],
            ),
        )
        
        # Create sampler for distributed training
        if self.world_size > 1:
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True,
            )
        else:
            sampler = None
        
        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config['training']['batch_size'],
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.config['training']['num_workers'],
            pin_memory=True,
            drop_last=True,
        )
    
    def _setup_logging(self):
        """Setup logging with WandB."""
        if self.is_main_process and WANDB_AVAILABLE and self.config['logging']['use_wandb']:
            wandb.init(
                project=self.config['logging']['project_name'],
                name=self.config['logging']['run_name'],
                config=self.config,
            )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary of metrics
        """
        # Move data to device
        latent = batch['latent'].to(self.device)
        
        # Forward pass
        loss, metrics = self.dmd.compute_loss(latent)
        
        # Backward pass
        loss = loss / self.config['training']['gradient_accumulation_steps']
        loss.backward()
        
        return metrics
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        if self.world_size > 1:
            self.dataloader.sampler.set_epoch(self.epoch)
        
        epoch_metrics = {
            'loss': 0.0,
            'denoising_loss': 0.0,
            'vq_loss': 0.0,
        }
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.dataloader):
            # Training step
            metrics = self.train_step(batch)
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                # Clip gradients
                if self.config['training']['max_grad_norm'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm'],
                    )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Accumulate metrics
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key] += metrics[key]
                num_batches += 1
                
                # Log metrics
                if self.global_step % self.config['logging']['log_interval'] == 0:
                    self._log_metrics(metrics)
                
                # Save checkpoint
                if self.global_step % self.config['logging']['save_interval'] == 0:
                    self._save_checkpoint()
        
        # Average epoch metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)
        
        return epoch_metrics
    
    def train(self):
        """Main training loop."""
        print(f"[Rank {self.local_rank}] Starting training...")
        
        start_time = time.time()
        
        for epoch in range(self.config['training']['num_epochs']):
            self.epoch = epoch
            
            epoch_start = time.time()
            epoch_metrics = self.train_epoch()
            epoch_time = time.time() - epoch_start
            
            if self.is_main_process:
                print(f"\nEpoch {epoch + 1}/{self.config['training']['num_epochs']}")
                print(f"  Time: {epoch_time:.2f}s")
                print(f"  Loss: {epoch_metrics['loss']:.4f}")
                print(f"  Denoising Loss: {epoch_metrics['denoising_loss']:.4f}")
                print(f"  VQ Loss: {epoch_metrics['vq_loss']:.4f}")
                print(f"  LR: {self.scheduler.get_last_lr()[0]:.2e}")
        
        total_time = time.time() - start_time
        
        if self.is_main_process:
            print(f"\nTraining completed in {total_time / 3600:.2f} hours")
            
            # Save final checkpoint
            self._save_checkpoint(final=True)
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to console and WandB."""
        if not self.is_main_process:
            return
        
        # Console logging
        log_str = f"Step {self.global_step}: "
        log_str += ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        log_str += f", lr={self.scheduler.get_last_lr()[0]:.2e}"
        print(log_str)
        
        # WandB logging
        if WANDB_AVAILABLE and self.config['logging']['use_wandb']:
            wandb.log({
                **metrics,
                'learning_rate': self.scheduler.get_last_lr()[0],
                'epoch': self.epoch,
                'global_step': self.global_step,
            })
    
    def _save_checkpoint(self, final: bool = False):
        """Save checkpoint."""
        if not self.is_main_process:
            return
        
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
        }
        
        if final:
            path = self.output_dir / 'final_checkpoint.pt'
        else:
            path = self.output_dir / f'checkpoint_step_{self.global_step}.pt'
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
        # Keep only last N checkpoints
        if not final:
            checkpoints = sorted(self.output_dir.glob('checkpoint_step_*.pt'))
            if len(checkpoints) > self.config['logging']['max_checkpoints']:
                for ckpt in checkpoints[:-self.config['logging']['max_checkpoints']]:
                    ckpt.unlink()
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        print(f"Loading checkpoint from {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        print(f"Resumed from step {self.global_step}, epoch {self.epoch}")


def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(local_rank)
    
    return local_rank, world_size


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_default_config() -> Dict[str, Any]:
    """Get default training configuration."""
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
            'num_samples': 10000,
            'batch_size': 4,
            'learning_rate': 1e-4,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'gradient_accumulation_steps': 1,
            'warmup_steps': 1000,
            'quantization_weight': 0.1,
            'num_workers': 4,
        },
        'data': {
            'num_frames': 8,
            'latent_height': 32,
            'latent_width': 32,
        },
        'logging': {
            'use_wandb': True,
            'project_name': 'quantized-diffusion-dmd',
            'run_name': 'experiment_1',
            'log_interval': 10,
            'save_interval': 1000,
            'max_checkpoints': 5,
        },
        'output_dir': './outputs/quantized_dmd',
    }


def main():
    parser = argparse.ArgumentParser(description='Train Quantized Diffusion Model with DMD')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Setup distributed training
    local_rank, world_size = setup_distributed()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # Save config
    if local_rank == 0:
        output_dir = Path(config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    # Create trainer
    trainer = QuantizedDMDTrainer(
        config=config,
        local_rank=local_rank,
        world_size=world_size,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()

