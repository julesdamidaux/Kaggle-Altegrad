#!/bin/bash
# Monitor training progress

LOG_FILE="/Data/hamzaazzouzi/Kaggle-Altegrad/training_log.txt"
CHECKPOINT_DIR="/Data/hamzaazzouzi/Kaggle-Altegrad/Hamza/checkpoints/qwen2.5-0.5b-graph-mlp-prompts"

echo "=================================="
echo "Training Monitor"
echo "=================================="
echo ""

# Check if training is running
if ps aux | grep -v grep | grep "finetune_llm_with_graph_encoder" > /dev/null; then
    echo "✓ Training process is RUNNING"
else
    echo "✗ Training process is NOT running"
fi

echo ""
echo "Latest training output:"
echo "---"
tail -30 "$LOG_FILE" 2>/dev/null || echo "No log file yet"
echo "---"

echo ""
echo "Checkpoints saved:"
if [ -d "$CHECKPOINT_DIR" ]; then
    ls -lh "$CHECKPOINT_DIR" 2>/dev/null | grep -E "^d|checkpoint" || echo "No checkpoints yet"
else
    echo "Checkpoint directory not created yet"
fi

echo ""
echo "To view live training:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To stop training:"
echo "  pkill -f finetune_llm_with_graph_encoder"
