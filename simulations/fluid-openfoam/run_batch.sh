#!/bin/bash
#
# This script orchestrates a batch run for the OpenFOAM solver.
# It loops through parameter files, generates a case for each, and then
# calls the single-purpose 'run.sh' script within that case directory.
#

set -e

# --- Path Setup ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

# --- Configuration ---
PYTHON_EXEC="python3"
PARAM_SETS_DIR="$SCRIPT_DIR/param_sets"
BASE_RUN_DIR="$SCRIPT_DIR/runs"
BASE_DATA_DIR="$PROJECT_ROOT/data/fluid-openfoam" # Final output location
GENERATE_CASE_SCRIPT="$SCRIPT_DIR/generate_case.py"
SINGLE_CASE_RUNNER="$SCRIPT_DIR/run.sh"
TIMEOUT_SECONDS="300s"
SENTINEL_FILE=".solver_ready"

# --- Main Loop ---
echo "SOLVER BATCH: Starting..."
PARAM_FILES=$(find "$PARAM_SETS_DIR" -name "case_*.yml" | sort)

if [ -z "$PARAM_FILES" ]; then
    echo "SOLVER BATCH: No parameter files found in $PARAM_SETS_DIR. Exiting."
    exit 1
fi

for param_file in $PARAM_FILES; do
    run_name=$(basename "$param_file" .yml | sed 's/case/run/')
    
    # Check if the final output file already exists
    final_output_path="$BASE_DATA_DIR/$run_name.npz"
    if [ -f "$final_output_path" ]; then
        echo "SOLVER BATCH: SKIPPING case '$run_name', output already exists."
        continue
    fi

    output_dir="$BASE_RUN_DIR/$run_name"
    
    echo "----------------------------------------------------"
    echo "SOLVER BATCH: Processing case '$run_name'"
    echo "----------------------------------------------------"

    # 1. Generate the case
    $PYTHON_EXEC "$GENERATE_CASE_SCRIPT" "$param_file" "$output_dir"

    # 2. Create sentinel file and run the single-case script
    touch "$output_dir/$SENTINEL_FILE"
    (
        cd "$output_dir"
        echo "SOLVER BATCH: Executing run.sh for '$run_name'..."
        timeout "$TIMEOUT_SECONDS" bash "$SINGLE_CASE_RUNNER" || if [ $? -eq 124 ]; then
            echo "SOLVER BATCH: Case '$run_name' timed out."
        else
            echo "SOLVER BATCH: Case '$run_name' failed. Check logs in '$output_dir'."
        fi
    )
    # 3. Clean up sentinel file
    rm -f "$output_dir/$SENTINEL_FILE"

    cd "$PROJECT_ROOT" # Return to the project root for the next loop iteration
done

echo "SOLVER BATCH: Finished."
