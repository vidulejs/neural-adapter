#!/bin/bash

if [ -z "$1" ] || ! [[ "$1" =~ ^[1-2]$ ]]; then
    echo "Usage: ./run.sh [1|2]"
    exit 1
fi

DIM=$1

# two directories up
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." &> /dev/null && pwd )"
cd "$PROJECT_ROOT"

echo "Starting datagen for ${DIM}D case..."
python -m precice_datagen.datagen.datagen $DIM