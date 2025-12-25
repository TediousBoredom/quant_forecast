#!/bin/bash
# 推理脚本 - Quantized Diffusion Model Inference
# Inference script for quantized diffusion model

set -e

echo "=========================================="
echo "Quantized Diffusion Model - Inference"
echo "=========================================="

# 检查参数
if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path> [options]"
    echo ""
    echo "Example:"
    echo "  $0 outputs/quantized_dmd_medium/final_checkpoint.pt"
    echo "  $0 outputs/quantized_dmd_medium/checkpoint_step_5000.pt --num_samples 10"
    exit 1
fi

CHECKPOINT=$1
shift

# 检查 checkpoint 是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"

# 默认参数
OUTPUT_DIR="outputs/inference_$(date +%Y%m%d_%H%M%S)"
NUM_SAMPLES=5
NUM_FRAMES=16
HEIGHT=256
WIDTH=256
NUM_STEPS=50
SEED=42

# 询问用户配置
echo ""
read -p "Number of samples to generate [default: $NUM_SAMPLES]: " input
NUM_SAMPLES=${input:-$NUM_SAMPLES}

read -p "Number of frames [default: $NUM_FRAMES]: " input
NUM_FRAMES=${input:-$NUM_FRAMES}

read -p "Number of inference steps [default: $NUM_STEPS]: " input
NUM_STEPS=${input:-$NUM_STEPS}

read -p "Enable visualization? [y/N]: " visualize
read -p "Analyze codebook usage? [y/N]: " analyze

# 构建命令
CMD="python inference_quantized.py \
    --checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --num_samples $NUM_SAMPLES \
    --num_frames $NUM_FRAMES \
    --height $HEIGHT \
    --width $WIDTH \
    --num_inference_steps $NUM_STEPS \
    --seed $SEED"

if [[ $visualize == "y" || $visualize == "Y" ]]; then
    CMD="$CMD --visualize"
fi

if [[ $analyze == "y" || $analyze == "Y" ]]; then
    CMD="$CMD --analyze_codebook"
fi

# 添加额外参数
CMD="$CMD $@"

echo ""
echo "Running inference with command:"
echo "$CMD"
echo ""

# 执行推理
eval $CMD

echo ""
echo "=========================================="
echo "Inference completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

