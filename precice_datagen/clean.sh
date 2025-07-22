#!/bin/bash
echo "Cleaning up preCICE, Python, and VTK temporary files..."

# directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

rm -rf "$DIR/precice-run"

find "$DIR" -type f -name "*.vtu" -delete
find "$DIR" -type f -name "*.pvtu" -delete
find "$DIR" -type f -name "*.series" -delete

echo "Cleanup complete."
