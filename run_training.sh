#!/bin/bash
# 快速启动脚本 - Quantized Diffusion Model Training
# Quick start script for training quantized diffusion model

set -e

echo "=========================================="
echo "Quantized Diffusion Model - Quick Start"
echo "=========================================="

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "Error: Python not found!"
    exit 1
fi

echo "Python version: $(python --version)"

# 创建必要的目录
echo ""
echo "Creating directories..."
mkdir -p configs
mkdir -p outputs
mkdir -p logs

# 生成配置文件
echo ""
echo "Generating configuration files..."
python config.py

# 检查 GPU
echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}') if torch.cuda.is_available() else None"

# 询问用户选择
echo ""
echo "Select training mode:"
echo "1) Small model (quick test, single GPU)"
echo "2) Medium model (standard training, single GPU)"
echo "3) Medium model (multi-GPU training)"
echo "4) DMD optimized (multi-GPU training)"
echo "5) Test model only (no training)"

read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "Starting small model training..."
        python train_quantized_dmd.py \
            --config configs/config_small.json \
            2>&1 | tee logs/train_small_$(date +%Y%m%d_%H%M%S).log
        ;;
    2)
        echo ""
        echo "Starting medium model training (single GPU)..."
        python train_quantized_dmd.py \
            --config configs/config_medium.json \
            2>&1 | tee logs/train_medium_$(date +%Y%m%d_%H%M%S).log
        ;;
    3)
        read -p "Enter number of GPUs [default: 8]: " num_gpus
        num_gpus=${num_gpus:-8}
        echo ""
        echo "Starting medium model training with $num_gpus GPUs..."
        torchrun --nproc_per_node=$num_gpus \
            train_quantized_dmd.py \
            --config configs/config_medium.json \
            2>&1 | tee logs/train_medium_multi_$(date +%Y%m%d_%H%M%S).log
        ;;
    4)
        read -p "Enter number of GPUs [default: 8]: " num_gpus
        num_gpus=${num_gpus:-8}
        echo ""
        echo "Starting DMD optimized training with $num_gpus GPUs..."
        torchrun --nproc_per_node=$num_gpus \
            train_quantized_dmd.py \
            --config configs/config_dmd_optimized.json \
            2>&1 | tee logs/train_dmd_$(date +%Y%m%d_%H%M%S).log
        ;;
    5)
        echo ""
        echo "Testing model..."
        python quantized_diffusion_model.py
        ;;
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="

