"""
Configuration for Graph Encoder + LLM Fine-tuning

This configuration file allows easy customization of:
- Graph encoder settings
- MLP projection architecture  
- Learnable prompt configuration
- LLM and LoRA settings
- Training hyperparameters
"""

# ============================================================================
# MODEL ARCHITECTURE SETTINGS
# ============================================================================

# LLM Configuration
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # or "Qwen/Qwen2.5-3B-Instruct"
CACHE_DIR = "./cache"

# Graph Encoder Configuration
GRAPH_ENCODER_CHECKPOINT = "/Data/hamzaazzouzi/Kaggle-Altegrad/Hamza/RAG+Predict/embeddings/encoder_checkpoints/best_encoder.pt"
GRAPH_HIDDEN_DIM = 256
GRAPH_LAYERS = 5
NODE_DIM = 9  # From x_map in data_utils.py
EDGE_DIM = 3  # From e_map in data_utils.py
FREEZE_GRAPH_ENCODER = True  # Set to False to fine-tune graph encoder

# MLP Projection Configuration
NUM_GRAPH_TOKENS = 4  # Number of tokens to represent the projected graph
# The MLP will project graph_encoding [256D] → [NUM_GRAPH_TOKENS × LLM_DIM]

# Learnable Prompts Configuration
NUM_LEARNABLE_PROMPTS = 8  # Number of trainable prefix tokens
# These tokens will be learned during training to help condition the LLM

# ============================================================================
# TRAINING SETTINGS
# ============================================================================

# Output Directory
OUTPUT_DIR = "./Hamza/checkpoints/qwen2.5-0.5b-graph-mlp-prompts"

# Data Paths
TRAIN_GRAPHS = "/Data/hamzaazzouzi/Kaggle-Altegrad/data_baseline/data/train_graphs.pkl"
VAL_GRAPHS = "/Data/hamzaazzouzi/Kaggle-Altegrad/data_baseline/data/validation_graphs.pkl"

# Training Hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS  # 16

LEARNING_RATE = 2e-4
WARMUP_STEPS = 200
MAX_LENGTH = 512

# Logging and Saving
LOGGING_STEPS = 10
SAVE_STEPS = 500
EVAL_STEPS = 500
SAVE_TOTAL_LIMIT = 5

# ============================================================================
# LORA CONFIGURATION
# ============================================================================

USE_LORA = True
LORA_R = 16  # LoRA rank
LORA_ALPHA = 32  # LoRA alpha (scaling factor)
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
    # Uncomment below to also apply LoRA to MLP layers (more parameters but better performance)
    # "gate_proj", "up_proj", "down_proj"
]

# ============================================================================
# QUANTIZATION AND OPTIMIZATION
# ============================================================================

USE_4BIT = True  # Use 4-bit quantization (saves memory)
USE_8BIT = False  # Alternative: 8-bit quantization

# Mixed Precision Training
FP16 = False  # Use FP16 (not recommended with 4-bit)
BF16 = True   # Use BF16 (recommended, more stable)

# Memory Optimization
USE_GRADIENT_CHECKPOINTING = True  # Trade compute for memory
MAX_GRAD_NORM = 1.0  # Gradient clipping

# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """You are a chemical annotation model trained for molecular graph captioning.

Your role is to generate ontology-style molecular descriptions similar to curated entries from chemical databases such as ChEBI or PubChem.

Hard constraints:
- Output a single paragraph in formal scientific English
- The paragraph must contain 3-4 complete sentences
- Do not use bullet points, lists, or line breaks
- Do not mention graphs, atoms, nodes, edges, SMILES, or machine learning concepts

Mandatory structure:
1. First sentence must begin: "The molecule is a ..." describing the chemical class
2. Include: "It has a role as a metabolite."
3. Include: "It is a <CLASS_1>, a <CLASS_2> and a <CLASS_3>."
4. When appropriate, include derivation: "It derives from <parent compound>."

Use dense chemical and ontological vocabulary. Prefer scaffold-level descriptions.
"""

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Device
DEVICE = "cuda"  # or "cpu"

# Random Seed
SEED = 42

# Evaluation Strategy
EVALUATION_STRATEGY = "steps"  # "steps" or "epoch"
LOAD_BEST_MODEL_AT_END = False  # Disabled due to custom save format
METRIC_FOR_BEST_MODEL = "loss"

# Weights & Biases (W&B) Logging
USE_WANDB = False
WANDB_PROJECT = "molecule-graph-llm"
WANDB_RUN_NAME = "qwen-graph-mlp-prompts"

# DataLoader Settings
DATALOADER_NUM_WORKERS = 4
DATALOADER_PIN_MEMORY = True

# ============================================================================
# ARCHITECTURE NOTES
# ============================================================================
"""
The model architecture is:

Input: Molecular Graph
    ↓
[1] Graph Encoder (MoleculeGINE)
    - Pre-trained, optionally frozen
    - Output: graph_encoding [batch, 256]
    ↓
[2] MLP Projector
    - Projects: [256] → [NUM_GRAPH_TOKENS × LLM_DIM]
    - Multi-layer with GELU, LayerNorm, Dropout
    - Trainable
    ↓
[3] Reshape to tokens: [batch, NUM_GRAPH_TOKENS, LLM_DIM]
    ↓
[4] Concatenate with Learnable Prompts
    - [batch, NUM_LEARNABLE_PROMPTS, LLM_DIM]
    - Trainable parameters
    ↓
[5] Concatenate with Text Embeddings
    - Full sequence: [prompts | graph_tokens | text_tokens]
    ↓
[6] LLM (with LoRA)
    - Fine-tuned with LoRA adapters
    - Generates molecule description
    ↓
Output: Description text

Training:
- Graph Encoder: Frozen (or fine-tuned if FREEZE_GRAPH_ENCODER=False)
- MLP Projector: Trained from scratch
- Learnable Prompts: Trained from scratch
- LLM: LoRA fine-tuning

This setup allows the model to:
1. Use pre-trained graph understanding
2. Learn optimal projection to LLM space
3. Learn task-specific prefix conditioning
4. Adapt LLM for molecule description generation
"""
