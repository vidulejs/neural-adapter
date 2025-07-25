#!/bin/bash
set -e

echo "Starting solver..."
cd "$(dirname "$0")"
python solver.py "$@"
