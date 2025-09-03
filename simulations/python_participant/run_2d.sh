#!/bin/bash
#
# This script runs the Python datagen participant for a single case.
# It expects to be run from the corresponding case directory in simulations/fluid-openfoam/runs/
#

set -e

# --- Path Setup ---
RUN_DIR=$(pwd)
# SCRIPT_DIR is the absolute path to the directory where this script is located.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

# --- Configuration ---
PYTHON_EXEC="python3"
DATAGEN_SCRIPT="$SCRIPT_DIR/datagen_2d.py"

# --- Argument Parsing ---
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path_to_param_file> <output_path_for_npz>"
    exit 1
fi

PARAM_FILE=$1
NPZ_OUTPUT_PATH=$2

# --- Main ---
echo "DATAGEN: Starting for case in $(basename "$RUN_DIR")"

$PYTHON_EXEC "$DATAGEN_SCRIPT" --params "$PARAM_FILE" --output-path "$NPZ_OUTPUT_PATH"

echo "DATAGEN: Finished."