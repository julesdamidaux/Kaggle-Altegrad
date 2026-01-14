#!/bin/bash
# Master script to run the complete RAG pipeline for molecular graph captioning
# Run this in the 'rag' tmux session

set -e  # Exit on error

echo "================================================================================================"
echo "MOLECULAR GRAPH CAPTIONING WITH RAG AND GRAPH ENCODINGS"
echo "================================================================================================"
echo ""
echo "This script will:"
echo "1. Train the graph encoder using contrastive learning"
echo "2. Encode all molecules and retrieve 5 closest descriptions for each"
echo "3. Fine-tune Qwen LLM with graph encodings and retrieved descriptions"
echo "4. Generate predictions on test set"
echo ""
echo "================================================================================================"
echo ""

# Set paths
WORKSPACE_ROOT="/Data/hamzaazzouzi/Kaggle-Altegrad"
cd "$WORKSPACE_ROOT"

# Create log directory
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/rag_pipeline_$TIMESTAMP.log"

echo "Logging to: $LOG_FILE"
echo ""

# Function to log and execute
run_step() {
    local step_num=$1
    local step_name=$2
    local command=$3
    
    echo ""
    echo "================================================================================================"
    echo "STEP $step_num: $step_name"
    echo "================================================================================================"
    echo "Command: $command"
    echo ""
    
    # Run and log
    if eval "$command 2>&1 | tee -a $LOG_FILE"; then
        echo ""
        echo "✓ STEP $step_num completed successfully!"
        echo ""
    else
        echo ""
        echo "✗ STEP $step_num failed! Check logs at $LOG_FILE"
        echo ""
        exit 1
    fi
}

# Step 1: Run complete RAG pipeline (train encoder, encode graphs, retrieve descriptions)
run_step 1 \
    "RAG Pipeline: Train Encoder + Encode Graphs + Retrieve Descriptions" \
    "python Hamza/RAG+Predict/complete_rag_pipeline.py"

# Step 2: Fine-tune Qwen with graph encodings and RAG
run_step 2 \
    "Fine-tune Qwen with Graph Encodings and RAG" \
    "python Hamza/train/finetune_qwen_with_rag.py"

# Step 3: Generate predictions on test set
run_step 3 \
    "Generate Predictions with RAG Model" \
    "python Hamza/train/inference_qwen_with_rag.py \
        --checkpoint ./Hamza/checkpoints/qwen2.5-rag/final \
        --data ./Hamza/RAG+Predict/embeddings/test_llm_data.pkl \
        --output ./Hamza/RAG+Predict/predictions/test_predictions_rag.csv \
        --batch_size 8 \
        --max_new_tokens 256"

echo ""
echo "================================================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "================================================================================================"
echo ""
echo "Output files:"
echo "  - Graph encoder: ./Hamza/RAG+Predict/embeddings/encoder_checkpoints/best_encoder.pt"
echo "  - Graph embeddings: ./Hamza/RAG+Predict/embeddings/train_embeddings.pkl"
echo "  - Retrieved descriptions: ./Hamza/RAG+Predict/embeddings/train_retrievals.pkl"
echo "  - LLM training data: ./Hamza/RAG+Predict/embeddings/train_llm_data.pkl"
echo "  - Fine-tuned model: ./Hamza/checkpoints/qwen2.5-rag/final/"
echo "  - Test predictions: ./Hamza/RAG+Predict/predictions/test_predictions_rag.csv"
echo ""
echo "Log file: $LOG_FILE"
echo ""
echo "================================================================================================"
