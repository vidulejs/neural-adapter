#!/bin/bash
set -e

echo "Starting datagen participant..."
cd "$(dirname "$0")"

# --- Configuration ---
# Set the starting epoch number here. This can be overridden by the START_EPOCH env var.
: "${start_epoch:=0}"
# The output path is now the second argument to the script, defaulting to "."
output_path="${2:-.}"
# -------------------

# Get the number of epochs to generate for this run
epochs_to_run=$(python3 -c "import json; print(json.load(open('config.json'))['datagen']['epochs'])")
end_epoch=$((start_epoch + epochs_to_run))

echo "Starting from epoch $start_epoch."
echo "Will run for $epochs_to_run epochs, until epoch $((end_epoch - 1))."
echo "Saving data to: $output_path"

for (( i=$start_epoch; i<$end_epoch; i++ ))
do
   echo "--- Starting Datagen Epoch $i ---"
   timeout 90s python3 datagen.py $1 --epoch $i --output-path "$output_path" || if [ $? -eq 124 ]; then echo "Datagen for epoch $i timed out. Continuing to next epoch."; else exit $?; fi
done
