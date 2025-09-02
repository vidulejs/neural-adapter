#!/bin/bash
# Usage: ./run_2d.sh <path_to_precice_config> <output_path_for_npz>

set -e

echo "--- Starting Datagen Participant (2D) ---"

# Check for correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path_to_precice_config> <output_path_for_npz>"
    exit 1
fi

CONFIG_FILE=$1
OUTPUT_PATH=$2
EPOCH_NUM=0

cd "$(dirname "$0")"

echo "Using preCICE config: $CONFIG_FILE"
echo "Saving data to: $OUTPUT_PATH"

timeout 180s /home/dan/venvs/torch/bin/python3 datagen_2d.py --config "$CONFIG_FILE" --epoch $EPOCH_NUM --output-path "$OUTPUT_PATH"

# Check the exit code of the python script
if [ $? -eq 124 ]; then
    echo "Datagen participant timed out after 180 seconds."
    exit 124
fi

echo "--- Datagen Participant Finished ---"
