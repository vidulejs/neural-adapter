#!/bin/bash
# Cleans all .yml files in this directory.

set -e

echo "Cleaning all .yml files in $(pwd)..."

# The script is inside the param_sets directory.
# We want to delete all .yml files here.
rm -f ./*.yml

echo "Cleanup complete."