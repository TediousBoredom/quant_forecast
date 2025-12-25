"""
测试脚本 - Quantized Diffusion Model
Test script for quantized diffusion model

运行各种测试以验证模型实现的正确性
"""

import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path

from quantized_diffusion_model import (
    VectorQuantizer,
    QuantizedDiffusionPredictor,
    QuantizedDiffusionDMD,
    TransformerBlock,
)
from utils import count_parameters, format_number, MetricsTracker


def test_vector_quantizer():
    """测试向量量化器"""
    print("\n" + "="*60)
    print("Testing VectorQuantizer...")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建 VQ
    vq = VectorQuantizer(
        num_embeddings=1024,
        embedding_dim=256,
    ).to(device)
    
    # 测试前向传播
    B, C, F, H, W = 2, 256, 4, 8, 8
    z = torch.randn(B, C, F, H, W, device=device)
    
    z_q, vq_loss, indices = vq(z)
    
    # 验证形状
    assert z_q.shape == z.shape, f"Shape mismatch: {z_q.shape} vs {z.shape}"
    assert indices.shape == (B, F, H, W), f"Indices shape mismatch: {indices.shape}"
    assert vq_loss.item() >= 0, "VQ loss should be non-negative"
    
    # 验证量化
    assert torch.allclose(z_q, z, atol=1.0), "Quantized output too different from input"
    
    # 测试编码/解码
    z_reconstructed = vq.get_codebook_entry(indices)
    assert z_reconstructed.shape == z.shape, "Reconstruction shape mismatch"
    
    print(f"✓ Input shape: {z.shape}")
    print(f"✓ Output shape: {z_q.shape}")
    print(f"✓ Indices shape: {indices.shape}")
    print(f"✓ VQ loss: {vq_loss.item():.4f}")
    print(f"✓ Unique codes used: {len(torch.unique(indices))}/{vq.num_embeddings}")
    print("✓ VectorQuantizer test passed!")


def test_transformer_block():
    """测试 Transformer 块"""
    print("\n" + "="*60)
    print("Testing TransformerBlock...")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建 Transformer 块
    block = TransformerBlock(
        hidden_dim=512,
        num_heads=8,
        dropout=0.1,
    ).to(device)
    
    # 测试前向传播
    B, F, N, D = 2, 4, 64, 512
    x = torch.randn(B, F, N, D, device=device)
    
    output = block(x)
    
    # 验证形状
    assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
    
    # 验证梯度流
    loss = output.mean()
    loss.backward()
    
    has_grad = any(p.grad is not None for p in block.parameters())
    assert has_grad, "No gradients computed"
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Parameters: {format_number(sum(p.numel() for p in block.parameters()))}")
    print("✓ TransformerBlock test passed!")


def test_quantized_diffusion_predictor():
    """测试量化扩散预测器"""
    print("\n" + "="*60)
    print("Testing QuantizedDiffusionPredictor...")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = QuantizedDiffusionPredictor(
        latent_channels=16,
        num_embeddings=1024,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        max_frames=32,
    ).to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Model parameters: {format_number(total_params)}")
    
    # 测试前向传播（带量化）
    B, C, F, H, W = 2, 16, 8, 16, 16
    x = torch.randn(B, C, F, H, W, device=device)
    timesteps = torch.randint(0, 1000, (B,), device=device)
    
    output = model(x, timesteps, return_quantized=True)
    
    # 验证输出
    assert 'pred' in output, "Missing 'pred' in output"
    assert 'vq_loss' in output, "Missing 'vq_loss' in output"
    assert 'indices' in output, "Missing 'indices' in output"
    
    pred = output['pred']
    vq_loss = output['vq_loss']
    indices = output['indices']
    
    assert pred.shape[0] == B, "Batch size mismatch"
    assert indices.shape == (B, F, H, W), f"Indices shape mismatch: {indices.shape}"
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Prediction shape: {pred.shape}")
    print(f"✓ Indices shape: {indices.shape}")
    print(f"✓ VQ loss: {vq_loss.item():.4f}")
    
    # 测试编码/解码
    indices_encoded = model.encode_to_indices(x, timesteps)
    assert indices_encoded.shape == indices.shape, "Encoding shape mismatch"
    
    decoded = model.decode_from_indices(indices)
    print(f"✓ Decoded shape: {decoded.shape}")
    
    # 测试梯度流
    loss = pred.mean() + vq_loss
    loss.backward()
    
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "No gradients computed"
    
    print("✓ QuantizedDiffusionPredictor test passed!")


def test_quantized_diffusion_dmd():
    """测试 DMD 训练包装器"""
    print("\n" + "="*60)
    print("Testing QuantizedDiffusionDMD...")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = QuantizedDiffusionPredictor(
        latent_channels=16,
        num_embeddings=512,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        max_frames=16,
    ).to(device)
    
    # 创建 DMD 包装器
    dmd = QuantizedDiffusionDMD(
        model=model,
        device=device,
        num_train_timesteps=1000,
        min_step=20,
        max_step=980,
        beta_schedule='linear',
        quantization_weight=0.1,
    )
    
    # 测试损失计算
    B, C, F, H, W = 2, 16, 4, 8, 8
    x_0 = torch.randn(B, C, F, H, W, device=device)
    
    loss, metrics = dmd.compute_loss(x_0)
    
    # 验证损失
    assert loss.item() >= 0, "Loss should be non-negative"
    assert 'denoising_loss' in metrics, "Missing denoising_loss"
    assert 'vq_loss' in metrics, "Missing vq_loss"
    
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Denoising loss: {metrics['denoising_loss']:.4f}")
    print(f"✓ VQ loss: {metrics['vq_loss']:.4f}")
    
    # 测试梯度
    loss.backward()
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "No gradients computed"
    
    # 测试采样
    print("\nTesting sampling...")
    model.eval()
    with torch.no_grad():
        samples = dmd.sample(
            shape=(1, C, F, H, W),
            num_inference_steps=10,
        )
    
    assert samples.shape == (1, C, F, H, W), f"Sample shape mismatch: {samples.shape}"
    print(f"✓ Sample shape: {samples.shape}")
    
    print("✓ QuantizedDiffusionDMD test passed!")


def test_training_step():
    """测试完整的训练步骤"""
    print("\n" + "="*60)
    print("Testing Training Step...")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = QuantizedDiffusionPredictor(
        latent_channels=16,
        num_embeddings=256,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    ).to(device)
    
    # 创建 DMD
    dmd = QuantizedDiffusionDMD(
        model=model,
        device=device,
    )
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 训练几步
    model.train()
    tracker = MetricsTracker()
    
    num_steps = 5
    B, C, F, H, W = 2, 16, 4, 8, 8
    
    print(f"\nRunning {num_steps} training steps...")
    start_time = time.time()
    
    for step in range(num_steps):
        # 生成随机数据
        x_0 = torch.randn(B, C, F, H, W, device=device)
        
        # 前向传播
        loss, metrics = dmd.compute_loss(x_0)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录指标
        tracker.update(metrics)
        
        if step % 2 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Completed {num_steps} steps in {elapsed:.2f}s")
    print(f"✓ Average metrics: {tracker}")
    print("✓ Training step test passed!")


def test_memory_usage():
    """测试内存使用"""
    print("\n" + "="*60)
    print("Testing Memory Usage...")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping memory test")
        return
    
    device = torch.device('cuda')
    
    # 清空缓存
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 创建模型
    model = QuantizedDiffusionPredictor(
        latent_channels=16,
        num_embeddings=1024,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
    ).to(device)
    
    model_memory = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"✓ Model memory: {model_memory:.2f} MB")
    
    # 前向传播
    B, C, F, H, W = 4, 16, 8, 16, 16
    x = torch.randn(B, C, F, H, W, device=device)
    timesteps = torch.randint(0, 1000, (B,), device=device)
    
    output = model(x, timesteps, return_quantized=True)
    
    forward_memory = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"✓ Forward pass memory: {forward_memory:.2f} MB")
    
    # 反向传播
    loss = output['pred'].mean() + output['vq_loss']
    loss.backward()
    
    backward_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"✓ Peak memory (with gradients): {backward_memory:.2f} MB")
    
    print("✓ Memory usage test passed!")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("Running All Tests for Quantized Diffusion Model")
    print("="*60)
    
    tests = [
        ("VectorQuantizer", test_vector_quantizer),
        ("TransformerBlock", test_transformer_block),
        ("QuantizedDiffusionPredictor", test_quantized_diffusion_predictor),
        ("QuantizedDiffusionDMD", test_quantized_diffusion_dmd),
        ("Training Step", test_training_step),
        ("Memory Usage", test_memory_usage),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n✗ {name} test FAILED!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠ {failed} test(s) failed")
    
    print("="*60)


if __name__ == '__main__':
    run_all_tests()

