#!/bin/bash
#
# Train Graph Encoder + LLM with MLP Projection and Learnable Prompts
#

set -e  # Exit on error

echo "=================================="
echo "Graph-LLM Training Script"
echo "=================================="
echo ""

# Activate conda environment
echo "Activating kaggle_altegrad environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate kaggle_altegrad

# Navigate to project root
cd /Data/hamzaazzouzi/Kaggle-Altegrad

echo ""
echo "Starting training..."
echo "This will:"
echo "  1. Load pre-trained graph encoder (frozen)"
echo "  2. Initialize MLP projector (trainable)"
echo "  3. Initialize learnable prompts (trainable)"
echo "  4. Fine-tune Qwen LLM with LoRA"
echo ""
echo "Output directory: Hamza/checkpoints/qwen2.5-0.5b-graph-mlp-prompts/"
echo ""

# Run training
python Hamza/train/finetune_llm_with_graph_encoder.py

echo ""
echo "=================================="
echo "Training Complete!"
echo "=================================="
echo ""
echo "Model saved to: Hamza/checkpoints/qwen2.5-0.5b-graph-mlp-prompts/"
echo ""
echo "To generate predictions, run:"
echo "  python Hamza/train/inference_graph_llm.py \\"
echo "    --checkpoint Hamza/checkpoints/qwen2.5-0.5b-graph-mlp-prompts/final \\"
echo "    --input data_baseline/data/test_graphs.pkl \\"
echo "    --output predictions_graph_llm.csv"
echo ""
