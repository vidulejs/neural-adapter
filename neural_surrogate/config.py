
# Data and paths

import os


_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_PATH, os.pardir))

DATA_DIR = "tutorials/partitioned-burgers-1d/solver-scipy-fvolumes/data-training"
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_DIR)

# Model architecture
INPUT_SIZE = 160 + 2 # + for ghost cells
HIDDEN_SIZE = 64 # num filters
OUTPUT_SIZE = 160

assert INPUT_SIZE >= OUTPUT_SIZE, "Input size must be greater or equal to output size."
assert (INPUT_SIZE - OUTPUT_SIZE) % 2 == 0, "Input and output sizes must differ by an even number (for ghost cells)."

NUM_RES_BLOCKS = 4
KERNEL_SIZE = 5

# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-4
SPLIT = 0.7  # Train/val split fraction
UNROLL_STEPS = 5