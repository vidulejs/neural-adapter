#!/bin/bash
# Cleans up all case directories within this directory.

set -e

echo "Cleaning all case directories inside $(pwd)..."

find . -mindepth 1 -not -name "$(basename "$0")" -exec rm -rf {} +

echo "Cleanup complete."