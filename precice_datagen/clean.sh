#!/bin/bash
echo "Cleaning up preCICE, Python, and VTK temporary files..."

# directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Source the helper functions
# shellcheck disable=SC1091
if [ -f "$DIR/tools/cleaning-tools.sh" ]; then
    . "$DIR/tools/cleaning-tools.sh"
else
    echo "Error: cleaning-tools.sh not found!"
    exit 1
fi

# Clean up precice-run directory
rm -rf "$DIR/precice-run"

# Clean OpenFOAM cases
for d in "$DIR"/*/; do
    if [ -d "$d/system" ] && [ -f "$d/system/controlDict" ]; then
        echo "Found OpenFOAM case in $d, cleaning..."
        clean_openfoam "$d"
    fi
done

# Find and delete VTK files
find "$DIR" -type f -name "*.vtu" -delete
find "$DIR" -type f -name "*.pvtu" -delete
find "$DIR" -type f -name "*.series" -delete

echo "Cleanup complete."
