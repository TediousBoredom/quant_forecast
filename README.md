# Quantized Diffusion Model for Video Generation
# 基于 Diffusion 的量化预测模型

这是一个基于 Diffusion 和 Distribution Matching Distillation (DMD) 的量化预测模型，用于高效的视频生成。

## 📋 目录结构

```
openveo3_dmd/0/
├── quantized_diffusion_model.py   # 核心模型实现
├── train_quantized_dmd.py         # 训练脚本
├── inference_quantized.py         # 推理脚本
├── config.py                       # 配置文件生成器
├── README.md                       # 本文件
└── configs/                        # 配置文件目录（自动生成）
```

## 🌟 主要特性

### 1. **向量量化 (Vector Quantization)**
- 使用 VQ-VAE 风格的离散编码本
- 支持 EMA 更新策略
- 可配置的编码本大小（4K-16K）

### 2. **Diffusion 去噪过程**
- 支持线性和余弦噪声调度
- DDIM 采样加速推理
- 可配置的时间步范围

### 3. **DMD 训练策略**
- Distribution Matching Distillation
- 高效的知识蒸馏
- 支持梯度累积和混合精度训练

### 4. **Transformer 架构**
- 多头自注意力机制
- 时空建模能力
- 位置编码和时间步嵌入

## 🚀 快速开始

### 1. 生成配置文件

```bash
cd /inspire/ssd/project/video-generation/public/openveo3/openveo3_dmd/0
python config.py
```

这将生成 4 个预设配置：
- `config_small.json` - 小模型，快速测试
- `config_medium.json` - 中等模型，标准训练
- `config_large.json` - 大模型，高质量生成
- `config_dmd_optimized.json` - DMD 优化配置

### 2. 训练模型

#### 单 GPU 训练
```bash
python train_quantized_dmd.py \
    --config configs/config_small.json
```

#### 多 GPU 训练（推荐）
```bash
torchrun --nproc_per_node=8 \
    train_quantized_dmd.py \
    --config configs/config_medium.json
```

#### 从检查点恢复训练
```bash
python train_quantized_dmd.py \
    --config configs/config_medium.json \
    --resume outputs/quantized_dmd_medium/checkpoint_step_5000.pt
```

### 3. 推理生成

#### 基本推理
```bash
python inference_quantized.py \
    --checkpoint outputs/quantized_dmd_medium/final_checkpoint.pt \
    --output_dir outputs/inference \
    --num_samples 5 \
    --num_frames 16 \
    --height 256 \
    --width 256
```

#### 带可视化的推理
```bash
python inference_quantized.py \
    --checkpoint outputs/quantized_dmd_medium/final_checkpoint.pt \
    --output_dir outputs/inference \
    --num_samples 5 \
    --visualize \
    --analyze_codebook
```

## 📊 模型架构

### 核心组件

1. **VectorQuantizer**
   - 编码本大小：4096-16384
   - 嵌入维度：256-768
   - EMA 衰减：0.99

2. **QuantizedDiffusionPredictor**
   - 输入投影：Conv3D
   - Transformer 块：6-24 层
   - 输出投影：Conv3D + 量化

3. **TransformerBlock**
   - 多头自注意力
   - 前馈网络（4x 扩展）
   - LayerNorm + 残差连接

### 训练流程

```
输入 x_0 (clean latent)
    ↓
采样时间步 t
    ↓
添加噪声 → x_t
    ↓
模型预测 → pred_x0
    ↓
向量量化 → z_q, indices
    ↓
计算损失：
  - 去噪损失：MSE(pred_x0, x_0)
  - VQ 损失：commitment + codebook
    ↓
反向传播 + 优化
```

## 🔧 配置说明

### 模型配置
```json
{
  "model": {
    "latent_channels": 16,        // 潜在空间通道数
    "num_embeddings": 8192,       // 编码本大小
    "embedding_dim": 512,         // 嵌入维度
    "hidden_dim": 1024,           // 隐藏层维度
    "num_layers": 12,             // Transformer 层数
    "num_heads": 16,              // 注意力头数
    "dropout": 0.1,               // Dropout 率
    "max_frames": 64              // 最大帧数
  }
}
```

### Diffusion 配置
```json
{
  "diffusion": {
    "num_train_timesteps": 1000,  // 训练时间步数
    "min_step": 20,                // 最小时间步
    "max_step": 980,               // 最大时间步
    "beta_schedule": "linear"      // 噪声调度类型
  }
}
```

### 训练配置
```json
{
  "training": {
    "num_epochs": 100,                    // 训练轮数
    "batch_size": 4,                      // 批次大小
    "learning_rate": 1e-4,                // 学习率
    "gradient_accumulation_steps": 2,     // 梯度累积步数
    "warmup_steps": 1000,                 // 预热步数
    "quantization_weight": 0.1            // VQ 损失权重
  }
}
```

## 📈 性能优化建议

### 1. 内存优化
- 使用梯度累积减少批次大小
- 启用混合精度训练（FP16/BF16）
- 使用 FSDP 进行大模型训练

### 2. 训练加速
- 增加 num_workers 提高数据加载速度
- 使用多 GPU 并行训练
- 调整 min_step/max_step 减少计算量

### 3. 质量提升
- 增大编码本大小（8K → 16K）
- 使用余弦噪声调度
- 降低 quantization_weight（0.1 → 0.05）
- 增加模型深度和宽度

## 🔬 实验建议

### 消融实验

1. **编码本大小**
   ```bash
   # 测试不同编码本大小：4096, 8192, 16384
   python train_quantized_dmd.py --config config_4k.json
   python train_quantized_dmd.py --config config_8k.json
   python train_quantized_dmd.py --config config_16k.json
   ```

2. **量化损失权重**
   ```python
   # 在 config 中修改
   "quantization_weight": [0.05, 0.1, 0.2, 0.5]
   ```

3. **时间步范围**
   ```python
   # 测试不同范围
   "min_step": [20, 50, 100]
   "max_step": [900, 950, 980]
   ```

## 📊 评估指标

模型训练过程中会记录以下指标：

- **loss**: 总损失
- **denoising_loss**: 去噪损失（MSE）
- **vq_loss**: 向量量化损失
- **codebook_usage**: 编码本使用率
- **perplexity**: 编码本困惑度

推理时可以分析：
- 编码本使用统计
- 生成质量（FVD, IS 等）
- 推理速度（FPS）

## 🐛 常见问题

### 1. OOM (Out of Memory)
```bash
# 解决方案：
# - 减小 batch_size
# - 增加 gradient_accumulation_steps
# - 减小模型尺寸（hidden_dim, num_layers）
# - 减小输入分辨率
```

### 2. 编码本崩溃
```python
# 症状：只使用少数几个编码
# 解决方案：
# - 降低 commitment_cost
# - 增加 EMA decay
# - 使用更大的编码本
# - 增加训练数据多样性
```

### 3. 训练不稳定
```python
# 解决方案：
# - 降低学习率
# - 增加 warmup_steps
# - 使用梯度裁剪
# - 检查数据归一化
```

## 📚 参考文献

1. **VQ-VAE**: Neural Discrete Representation Learning (van den Oord et al., 2017)
2. **DDPM**: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
3. **DDIM**: Denoising Diffusion Implicit Models (Song et al., 2020)
4. **DMD**: Distribution Matching Distillation (Yin et al., 2024)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目遵循 MIT 许可证。

## 📧 联系方式

如有问题，请联系项目维护者。

---

**注意**: 这是一个研究性质的实现，用于探索量化扩散模型在视频生成中的应用。在生产环境使用前，请进行充分的测试和验证。

