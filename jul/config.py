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
# LLM Decoder
LLM_MODEL_NAME = "Qwen/Qwen3-4B"  # Upgraded to 4B model
USE_LORA = True  # Use LoRA for efficient fine-tuning
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# =========================================================
# Training Configuration
# =========================================================
# TRAIN_STAGE is now automated.
EPOCHS_STAGE1 = 4  # Phase 1: Frozen LLM (Q-Former only)
EPOCHS_STAGE2 = 4  # Phase 2: Joint Training (LoRA)

BATCH_SIZE = 4  # Reduced to 4 for 4B model on 24GB GPU
GRADIENT_ACCUMULATION_STEPS = 32  # Increased to maintain effective batch size

# Learning Rates
LEARNING_RATE_STAGE1_HEAD = 1e-4  # Phase 1: Q-Former + Projector
LEARNING_RATE_STAGE2_HEAD = 5e-5  # Phase 2: Q-Former + Projector
LEARNING_RATE_STAGE2_LORA = 1e-5  # Phase 2: LLM LoRA

WARMUP_STEPS = 120  # 10% of total training steps (~2420)
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
GENERATION_MAX_NEW_TOKENS = 256  # Maximum number of new tokens to generate
GENERATION_MIN_NEW_TOKENS = 30  # Minimum tokens to prevent truncation
GENERATION_NUM_BEAMS = 1  # Reverted to 1 (sampling) to save memory
GENERATION_TEMPERATURE = 0.2  # Low temperature for stability
GENERATION_TOP_P = 0.9  # Nucleus sampling
GENERATION_REPETITION_PENALTY = 1.2  # Increased to prevent loops
GENERATION_NO_REPEAT_NGRAM_SIZE = 0  # Standard value
GENERATION_LENGTH_PENALTY = 1.0  # Neutral length penalty

# Prompt Engineering
USE_FEW_SHOT_PROMPTING = True  

