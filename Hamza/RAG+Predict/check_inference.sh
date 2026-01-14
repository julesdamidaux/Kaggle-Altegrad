#!/bin/bash
# Monitor inference progress

echo "=================================="
echo "Inference Monitor"
echo "=================================="
echo ""

# Check if process is running
PID=$(ps aux | grep "predict_with_graph_llm.py" | grep python | grep -v grep | awk '{print $2}' | tail -1)

if [ -z "$PID" ]; then
    echo "✗ Inference process is NOT running"
    echo ""
    echo "Start it with:"
    echo "  cd /Data/hamzaazzouzi/Kaggle-Altegrad/Hamza/RAG+Predict"
    echo "  ./run_inference.sh"
    exit 1
else
    echo "✓ Inference process is RUNNING (PID: $PID)"
    
    # Check how long it's been running
    START_TIME=$(ps -p $PID -o lstart= 2>/dev/null)
    if [ ! -z "$START_TIME" ]; then
        echo "  Started: $START_TIME"
    fi
    
    # Check CPU usage
    CPU=$(ps -p $PID -o %cpu= 2>/dev/null)
    MEM=$(ps -p $PID -o %mem= 2>/dev/null)
    echo "  CPU: ${CPU}%  |  Memory: ${MEM}%"
fi

echo ""
echo "Checkpoint: checkpoint-1938 (final trained model)"
echo "Output file: predictions_graph_llm.csv"
echo ""

# Check if output file exists
if [ -f "/Data/hamzaazzouzi/Kaggle-Altegrad/predictions_graph_llm.csv" ]; then
    LINES=$(wc -l < /Data/hamzaazzouzi/Kaggle-Altegrad/predictions_graph_llm.csv)
    COMPLETED=$((LINES - 1))  # Subtract header
    if [ $COMPLETED -gt 0 ]; then
        PERCENT=$((COMPLETED * 100 / 1000))
        echo "Progress: $COMPLETED / 1000 graphs ($PERCENT%)"
        echo ""
        
        # Estimate time remaining
        if [ $COMPLETED -gt 10 ]; then
            AVG_TIME=7  # seconds per graph
            REMAINING=$((1000 - COMPLETED))
            TIME_LEFT=$((REMAINING * AVG_TIME / 60))
            echo "Estimated time remaining: ~${TIME_LEFT} minutes"
        fi
    else
        echo "Status: Initializing..."
    fi
else
    echo "Status: Starting up (no predictions file yet)"
fi

echo ""
echo "To view sample predictions:"
echo "  head -5 /Data/hamzaazzouzi/Kaggle-Altegrad/predictions_graph_llm.csv"
echo ""
echo "To stop inference:"
echo "  kill $PID"
