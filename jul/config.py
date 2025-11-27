"""Configuration for Graph-to-Text model."""

import torch

# =========================================================
# Data Configuration
# =========================================================
TRAIN_GRAPHS = "../data_baseline/data/train_graphs.pkl"
VAL_GRAPHS = "../data_baseline/data/validation_graphs.pkl"
TEST_GRAPHS = "../data_baseline/data/test_graphs.pkl"

# =========================================================
# Model Configuration
# =========================================================
# Graph Encoder: Pretrained Graphormer (FROZEN)
GRAPHORMER_MODEL_NAME = "clefourrier/graphormer-base-pcqm4mv2"
GRAPH_HIDDEN_DIM = 768  # Graphormer output dimension

# Q-Former (Query Transformer)
NUM_QUERY_TOKENS = 32  # Learnable queries to extract graph features #32
QFORMER_HIDDEN_DIM = 768 
QFORMER_NUM_LAYERS = 6 # 6
QFORMER_NUM_HEADS = 8  # 8

# LLM Decoder
LLM_MODEL_NAME = "Qwen/Qwen3-1.7B-Base"  # Small model for 24GB GPU
USE_LORA = True  # Use LoRA for efficient fine-tuning
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# =========================================================
# Training Configuration
# =========================================================
BATCH_SIZE = 8  # Small batch size for memory efficiency
GRADIENT_ACCUMULATION_STEPS = 8  
MAX_EPOCHS = 10  # Increased for better convergence
LEARNING_RATE = 2e-5
# WARMUP_STEPS = 500
MAX_LENGTH = 256  # Maximum caption length

# Optimization
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# Paths
# =========================================================
CHECKPOINT_DIR = "./checkpoints"
OUTPUT_DIR = "./outputs"
LOG_DIR = "./logs"

# =========================================================
# Inference Configuration
# =========================================================
GENERATION_MAX_NEW_TOKENS = 200  # Maximum number of new tokens to generate
GENERATION_MIN_NEW_TOKENS = 30  # Minimum tokens to prevent truncation
GENERATION_NUM_BEAMS = 1  # Use sampling instead of beam search
GENERATION_TEMPERATURE = 0.7  # Standard temperature for generation
GENERATION_TOP_P = 0.9  # Nucleus sampling
GENERATION_REPETITION_PENALTY = 1.1  # Mild penalty
GENERATION_NO_REPEAT_NGRAM_SIZE = 0  # Disable hard n-gram blocking
GENERATION_LENGTH_PENALTY = 1.2  # Encourage longer, complete descriptions

# Prompt Engineering
USE_FEW_SHOT_PROMPTING = True  # Enable few-shot learning with examples (increases inference time)

