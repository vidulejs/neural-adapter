#!/bin/bash
#
# This script runs the Python parameter set generator with specified arguments.
#

set -e

START_NUM=50

NUM_CASES=50

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PYTHON_SCRIPT="$SCRIPT_DIR/generate_paramsets.py"

echo "Starting parameter generation..."
echo " - Start case number: $START_NUM"
echo " - Number of cases to generate: $NUM_CASES"

python3 "$PYTHON_SCRIPT" --start-num "$START_NUM" --num-cases "$NUM_CASES"

echo "Parameter generation finished."
