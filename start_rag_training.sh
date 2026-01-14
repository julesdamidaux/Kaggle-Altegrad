#!/bin/bash
# Helper script to run the RAG pipeline in the tmux session

echo "Starting RAG pipeline in tmux session 'rag'..."
echo ""

# Check if tmux session exists
if ! tmux has-session -t rag 2>/dev/null; then
    echo "Creating tmux session 'rag'..."
    tmux new-session -d -s rag
fi

# Send commands to the tmux session
tmux send-keys -t rag "cd /Data/hamzaazzouzi/Kaggle-Altegrad" C-m
tmux send-keys -t rag "clear" C-m
tmux send-keys -t rag "./run_complete_rag_pipeline.sh" C-m

echo "✓ Pipeline started in tmux session 'rag'"
echo ""
echo "To view progress:"
echo "  tmux attach -t rag"
echo ""
echo "To detach from tmux (while inside):"
echo "  Press Ctrl+b, then d"
echo ""
echo "To view logs:"
echo "  tail -f logs/rag_pipeline_*.log"
echo ""
