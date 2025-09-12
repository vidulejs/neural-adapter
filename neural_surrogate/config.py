
# Data and paths

import os


_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_PATH, os.pardir))

DATA_DIR = "data/solver-nutils/128_grid"
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_DIR)

# Model architecture
INPUT_SIZE = 128
HIDDEN_SIZE = 256
OUTPUT_SIZE = 128

# Training hyperparameters
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-6
SPLIT = 0.5  # Train/val split fraction
UNROLL_STEPS = 64
