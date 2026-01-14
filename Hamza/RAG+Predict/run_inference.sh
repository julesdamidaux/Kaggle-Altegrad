#!/bin/bash
# Run inference in the background with nohup

cd /Data/hamzaazzouzi/Kaggle-Altegrad

echo "Starting inference in background..."
echo "This will take approximately 2 hours for 1000 test graphs"
echo ""

nohup conda run -n kaggle_altegrad python Hamza/train/predict_with_graph_llm.py > inference_output.log 2>&1 &

PID=$!
echo "Process ID: $PID"
echo ""
echo "Monitor progress with:"
echo "  tail -f inference_output.log"
echo ""
echo "Check if running:"
echo "  ps aux | grep $PID"
echo ""
echo "Stop if needed:"
echo "  kill $PID"
