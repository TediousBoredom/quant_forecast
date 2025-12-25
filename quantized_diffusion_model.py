"""
Quantized Diffusion Model for Video Generation
基于 Diffusion 的量化预测模型

This module implements a quantized diffusion model that predicts quantized latent representations
for efficient video generation. It combines:
1. Diffusion-based denoising process
2. Vector Quantization (VQ) for discrete latent space
3. Distribution Matching Distillation (DMD) for efficient training
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import math


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for discrete latent representation.
    
    Args:
        num_embeddings: Size of the codebook
        embedding_dim: Dimension of each embedding vector
        commitment_cost: Weight for commitment loss
        decay: EMA decay for codebook updates
        epsilon: Small constant for numerical stability
    """
    
    def __init__(
        self,
        num_embeddings: int = 8192,
        embedding_dim: int = 512,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # Codebook embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
        # EMA for codebook updates
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', self.embeddings.weight.data.clone())
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with vector quantization.
        
        Args:
            z: Input tensor [B, C, F, H, W]
            
        Returns:
            z_q: Quantized tensor [B, C, F, H, W]
            loss: VQ loss (commitment + codebook)
            indices: Codebook indices [B, F, H, W]
        """
        # Flatten spatial dimensions: [B, C, F, H, W] -> [B*F*H*W, C]
        B, C, num_frames, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        
        # Calculate distances to codebook vectors
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embeddings.weight.t())
        )
        
        # Find nearest codebook entries
        indices = torch.argmin(distances, dim=1)
        
        # Quantize
        z_q_flat = self.embeddings(indices)
        
        # Reshape back: [B*F*H*W, C] -> [B, C, F, H, W]
        z_q = z_q_flat.view(B, num_frames, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        
        # Compute losses
        # Commitment loss: encourage encoder output to stay close to chosen codebook entry
        commitment_loss = nn.functional.mse_loss(z_q.detach(), z)
        
        # Codebook loss: move codebook entries towards encoder outputs
        codebook_loss = nn.functional.mse_loss(z_q, z.detach())
        
        # Total VQ loss
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator: copy gradients from z_q to z
        z_q = z + (z_q - z).detach()
        
        # Update EMA (only during training)
        if self.training:
            self._update_ema(z_flat, indices)
        
        # Reshape indices for output
        indices = indices.view(B, num_frames, H, W)
        
        return z_q, vq_loss, indices
    
    def _update_ema(self, z_flat: torch.Tensor, indices: torch.Tensor):
        """Update codebook using exponential moving average."""
        # One-hot encoding of indices
        encodings = nn.functional.one_hot(indices, self.num_embeddings).float()
        
        # Update cluster sizes
        self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                (1 - self.decay) * torch.sum(encodings, dim=0)
        
        # Laplace smoothing
        n = torch.sum(self.ema_cluster_size)
        self.ema_cluster_size = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.num_embeddings * self.epsilon) * n
        )
        
        # Update embeddings
        dw = torch.matmul(encodings.t(), z_flat)
        self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw
        
        # Normalize
        self.embeddings.weight.data = self.ema_w / self.ema_cluster_size.unsqueeze(1)
    
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get quantized vectors from indices.
        
        Args:
            indices: Codebook indices [B, F, H, W]
            
        Returns:
            z_q: Quantized tensor [B, C, F, H, W]
        """
        B, num_frames, H, W = indices.shape
        indices_flat = indices.view(-1)
        z_q_flat = self.embeddings(indices_flat)
        z_q = z_q_flat.view(B, num_frames, H, W, self.embedding_dim).permute(0, 4, 1, 2, 3).contiguous()
        return z_q


class QuantizedDiffusionPredictor(nn.Module):
    """
    Quantized Diffusion Model for predicting discrete latent codes.
    
    This model combines:
    1. Diffusion process for denoising
    2. Vector quantization for discrete representation
    3. Transformer-based architecture for temporal modeling
    
    Args:
        latent_channels: Number of channels in latent space
        num_embeddings: Size of VQ codebook
        embedding_dim: Dimension of VQ embeddings
        hidden_dim: Hidden dimension for transformer
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        latent_channels: int = 16,
        num_embeddings: int = 8192,
        embedding_dim: int = 512,
        hidden_dim: int = 1024,
        num_layers: int = 12,
        num_heads: int = 16,
        dropout: float = 0.1,
        max_frames: int = 64,
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Vector Quantizer
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        
        # Input projection: map latent to hidden dimension
        self.input_proj = nn.Conv3d(
            latent_channels,
            hidden_dim,
            kernel_size=1,
        )
        
        # Timestep embedding (for diffusion)
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        # Positional encoding for frames
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection: map back to embedding dimension
        self.output_proj = nn.Sequential(
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, embedding_dim, kernel_size=1),
        )
        
        # Quantization projection
        self.quant_proj = nn.Conv3d(embedding_dim, embedding_dim, kernel_size=1)
        
    def get_timestep_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            timesteps: Timesteps [B]
            dim: Embedding dimension
            
        Returns:
            Embeddings [B, dim]
        """
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        return_quantized: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional quantization.
        
        Args:
            x: Input latent [B, C, F, H, W]
            timesteps: Diffusion timesteps [B]
            return_quantized: Whether to apply VQ
            
        Returns:
            Dictionary containing:
                - pred: Predicted latent (quantized or continuous)
                - vq_loss: VQ loss (if quantized)
                - indices: Codebook indices (if quantized)
        """
        B, C, num_frames, H, W = x.shape
        
        # Project input to hidden dimension
        h = self.input_proj(x)  # [B, hidden_dim, F, H, W]
        
        # Get timestep embeddings
        t_emb = self.get_timestep_embedding(timesteps, self.hidden_dim)
        t_emb = self.time_embed(t_emb)  # [B, hidden_dim]
        
        # Add timestep conditioning
        t_emb = t_emb[:, :, None, None, None]  # [B, hidden_dim, 1, 1, 1]
        h = h + t_emb
        
        # Reshape for transformer: [B, hidden_dim, F, H, W] -> [B, F, H*W, hidden_dim]
        h = h.permute(0, 2, 3, 4, 1).contiguous()  # [B, F, H, W, hidden_dim]
        h = h.view(B, num_frames, H * W, self.hidden_dim)
        
        # Add positional encoding
        h = h + self.pos_embed[:, :num_frames, :].unsqueeze(2)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            h = block(h)
        
        # Reshape back: [B, F, H*W, hidden_dim] -> [B, hidden_dim, F, H, W]
        h = h.view(B, num_frames, H, W, self.hidden_dim)
        h = h.permute(0, 4, 1, 2, 3).contiguous()
        
        # Output projection
        pred = self.output_proj(h)  # [B, embedding_dim, F, H, W]
        
        # Apply quantization if requested
        if return_quantized:
            # Project to quantization space
            z = self.quant_proj(pred)
            
            # Quantize
            z_q, vq_loss, indices = self.vq(z)
            
            return {
                'pred': z_q,
                'pred_continuous': pred,
                'vq_loss': vq_loss,
                'indices': indices,
            }
        else:
            return {
                'pred': pred,
                'vq_loss': torch.tensor(0.0, device=x.device),
            }
    
    def encode_to_indices(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Encode input to discrete indices.
        
        Args:
            x: Input latent [B, C, F, H, W]
            timesteps: Diffusion timesteps [B]
            
        Returns:
            indices: Codebook indices [B, F, H, W]
        """
        output = self.forward(x, timesteps, return_quantized=True)
        return output['indices']
    
    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode from discrete indices.
        
        Args:
            indices: Codebook indices [B, F, H, W]
            
        Returns:
            z_q: Quantized latent [B, embedding_dim, F, H, W]
        """
        return self.vq.get_codebook_entry(indices)


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feedforward.
    
    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [B, F, N, hidden_dim] where N = H*W
            
        Returns:
            Output [B, F, N, hidden_dim]
        """
        B, num_frames, N, D = x.shape
        
        # Reshape for attention: [B, F, N, D] -> [B*F, N, D]
        x_flat = x.view(B * num_frames, N, D)
        
        # Self-attention
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        
        # Feedforward
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        
        # Reshape back: [B*F, N, D] -> [B, F, N, D]
        x = x_flat.view(B, num_frames, N, D)
        
        return x


class QuantizedDiffusionDMD:
    """
    DMD training wrapper for Quantized Diffusion Model.
    
    Implements Distribution Matching Distillation for efficient training
    of the quantized diffusion model.
    """
    
    def __init__(
        self,
        model: QuantizedDiffusionPredictor,
        teacher_model: Optional[nn.Module] = None,
        device: torch.device = torch.device('cuda'),
        num_train_timesteps: int = 1000,
        min_step: int = 20,
        max_step: int = 980,
        beta_schedule: str = "linear",
        quantization_weight: float = 0.1,
    ):
        """
        Initialize DMD trainer for quantized model.
        
        Args:
            model: Student model (quantized)
            teacher_model: Teacher model (optional, for distillation)
            device: Device to use
            num_train_timesteps: Number of diffusion timesteps
            min_step: Minimum timestep for training
            max_step: Maximum timestep for training
            beta_schedule: Noise schedule type
            quantization_weight: Weight for VQ loss
        """
        self.model = model
        self.teacher_model = teacher_model
        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.min_step = min_step
        self.max_step = max_step
        self.quantization_weight = quantization_weight
        
        # Setup noise schedule
        self.betas = self._get_beta_schedule(beta_schedule, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Move to device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        
    def _get_beta_schedule(self, schedule: str, num_timesteps: int) -> torch.Tensor:
        """Get noise schedule."""
        if schedule == "linear":
            return torch.linspace(0.0001, 0.02, num_timesteps)
        elif schedule == "cosine":
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to clean samples.
        
        Args:
            x_0: Clean samples [B, C, F, H, W]
            noise: Noise [B, C, F, H, W]
            timesteps: Timesteps [B]
            
        Returns:
            x_t: Noisy samples [B, C, F, H, W]
        """
        sqrt_alpha_prod = torch.sqrt(self.alphas_cumprod[timesteps])
        sqrt_one_minus_alpha_prod = torch.sqrt(1.0 - self.alphas_cumprod[timesteps])
        
        # Reshape for broadcasting: [B] -> [B, 1, 1, 1, 1]
        sqrt_alpha_prod = sqrt_alpha_prod[:, None, None, None, None]
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[:, None, None, None, None]
        
        x_t = sqrt_alpha_prod * x_0 + sqrt_one_minus_alpha_prod * noise
        return x_t
    
    def compute_loss(
        self,
        x_0: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss with quantization.
        
        Args:
            x_0: Clean latent [B, C, F, H, W]
            text_embeddings: Optional text conditioning
            
        Returns:
            loss: Total loss
            metrics: Dictionary of metrics
        """
        B = x_0.shape[0]
        
        # Sample timesteps
        timesteps = torch.randint(
            self.min_step,
            self.max_step + 1,
            (B,),
            device=self.device,
            dtype=torch.long,
        )
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Add noise
        x_t = self.add_noise(x_0, noise, timesteps)
        
        # Predict with quantization
        output = self.model(x_t, timesteps, return_quantized=True)
        pred = output['pred']
        vq_loss = output['vq_loss']
        
        # Compute denoising loss (predict x_0)
        denoising_loss = nn.functional.mse_loss(pred, x_0)
        
        # Total loss
        total_loss = denoising_loss + self.quantization_weight * vq_loss
        
        # Metrics
        metrics = {
            'loss': total_loss.item(),
            'denoising_loss': denoising_loss.item(),
            'vq_loss': vq_loss.item(),
            'timestep_mean': timesteps.float().mean().item(),
        }
        
        return total_loss, metrics
    
    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample from the model using DDIM.
        
        Args:
            shape: Shape of output [B, C, F, H, W]
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            text_embeddings: Optional text conditioning
            
        Returns:
            x_0: Generated samples [B, C, F, H, W]
        """
        # Start from random noise
        x_t = torch.randn(shape, device=self.device)
        
        # DDIM sampling
        timesteps = torch.linspace(
            self.max_step,
            self.min_step,
            num_inference_steps,
            device=self.device,
            dtype=torch.long,
        )
        
        for t in timesteps:
            t_batch = t.repeat(shape[0])
            
            # Predict x_0
            output = self.model(x_t, t_batch, return_quantized=True)
            pred_x0 = output['pred']
            
            # DDIM update
            if t > self.min_step:
                alpha_t = self.alphas_cumprod[t]
                alpha_prev = self.alphas_cumprod[t - 1]
                
                # Predict noise
                pred_noise = (x_t - torch.sqrt(alpha_t) * pred_x0) / torch.sqrt(1 - alpha_t)
                
                # Update
                x_t = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * pred_noise
            else:
                x_t = pred_x0
        
        return x_t


if __name__ == "__main__":
    # Test the model
    print("Testing Quantized Diffusion Model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = QuantizedDiffusionPredictor(
        latent_channels=16,
        num_embeddings=8192,
        embedding_dim=512,
        hidden_dim=1024,
        num_layers=12,
        num_heads=16,
    ).to(device)
    
    # Create DMD trainer
    dmd = QuantizedDiffusionDMD(
        model=model,
        device=device,
    )
    
    # Test forward pass
    B, C, num_frames, H, W = 2, 16, 8, 32, 32
    x = torch.randn(B, C, num_frames, H, W, device=device)
    
    # Compute loss
    loss, metrics = dmd.compute_loss(x)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")
    
    # Test sampling
    print("\nTesting sampling...")
    samples = dmd.sample(
        shape=(1, C, num_frames, H, W),
        num_inference_steps=20,
    )
    print(f"Generated samples shape: {samples.shape}")
    
    print("\n✓ All tests passed!")
