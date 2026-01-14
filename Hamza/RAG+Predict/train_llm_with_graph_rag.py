#!/usr/bin/env python3
"""
Master script to train LLM with graph encodings and 5NN RAG retrieval.

Workflow:
1. Load graph encodings (already computed)
2. Load retrieved 5NN descriptions (already computed)
3. Fine-tune Qwen2.5 LLM with:
   - Graph encodings (projected to token space)
   - 5NN descriptions in system prompt
   - Target descriptions as labels
"""

import os
import sys

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
hamza_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(hamza_dir)
sys.path.insert(0, project_root)

from Hamza.train.finetune_qwen_with_rag import main

if __name__ == "__main__":
    print("="*80)
    print("TRAINING LLM WITH GRAPH ENCODINGS + 5NN RAG")
    print("="*80)
    print("\nWorkflow:")
    print("1. Graph encoder: Already trained ✓")
    print("2. Graph embeddings: Already computed ✓")
    print("3. 5NN Retrieval: Already computed ✓")
    print("4. LLM Fine-tuning: Starting now...")
    print("\n" + "="*80)
    
    main()
