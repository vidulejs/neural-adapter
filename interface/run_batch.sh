#!/bin/bash
#
# This script runs the Python datagen participant for a batch of cases.
# It finds all parameter files in the corresponding solver directory, waits
# for a sentinel file to be created by the solver script, and then calls
# the single-purpose 'run_2d.sh' script within each case directory.
#

set -e

# --- Path Setup ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

# --- Configuration ---
SOLVER_DIR="$PROJECT_ROOT/simulations/fluid-openfoam"
PARAM_SETS_DIR="$SOLVER_DIR/param_sets"
BASE_RUN_DIR="$SOLVER_DIR/runs" # Where to look for the generated cases
BASE_OUTPUT_DIR="$PROJECT_ROOT/data/fluid-openfoam" # Where to save the final .npz data
SINGLE_CASE_RUNNER="$SCRIPT_DIR/run_2d.sh"
TIMEOUT_SECONDS="310s" # Slightly longer than solver timeout
SENTINEL_FILE=".solver_ready"

# --- Main Loop ---
echo "DATAGEN BATCH: Starting..."
PARAM_FILES=$(find "$PARAM_SETS_DIR" -name "case_*.yml" | sort)

if [ -z "$PARAM_FILES" ]; then
    echo "DATAGEN BATCH: No parameter files found in $PARAM_SETS_DIR. Exiting."
    exit 1
fi

# Ensure the final data output directory exists
mkdir -p "$BASE_OUTPUT_DIR"

for param_file in $PARAM_FILES; do
    run_name=$(basename "$param_file" .yml | sed 's/case/run/')
    
    # Check if the final output file already exists
    npz_output_path="$BASE_OUTPUT_DIR/$run_name.npz"
    if [ -f "$npz_output_path" ]; then
        echo "DATAGEN BATCH: SKIPPING case '$run_name', output already exists."
        continue
    fi

    run_dir="$BASE_RUN_DIR/$run_name"
    sentinel_path="$run_dir/$SENTINEL_FILE"
    
    echo "----------------------------------------------------"
    echo "DATAGEN BATCH: Processing case '$run_name'"
    echo "----------------------------------------------------"

    # Wait for the sentinel file to be created by the solver script
    while [ ! -f "$sentinel_path" ]; do
        echo "DATAGEN BATCH: Waiting for solver to be ready for '$run_name'..."
        sleep 2
    done

    # Run the datagen script inside the case directory
    (
        cd "$run_dir"
        echo "DATAGEN BATCH: Executing run_2d.sh for '$run_name'..."
        timeout "$TIMEOUT_SECONDS" bash "$SINGLE_CASE_RUNNER" "$param_file" "$npz_output_path" || if [ $? -eq 124 ]; then
            echo "DATAGEN BATCH: Case '$run_name' timed out."
        else
            echo "DATAGEN BATCH: Case '$run_name' failed."
        fi
    )
done

echo "DATAGEN BATCH: Finished."
