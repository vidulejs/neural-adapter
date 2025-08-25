#!/bin/bash

# Get the number of epochs from config.json
cd "$(dirname "$0")"
epochs=$(python3 -c "import json; print(json.load(open('../datagen/config.json'))['datagen']['epochs'])")

for (( i=0; i<epochs; i++ ))
do
   echo "--- Starting Solver Epoch $i ---"
   timeout 60s python3 solver.py $1 || if [ $? -eq 124 ]; then echo "Solver for epoch $i timed out. Continuing to next epoch."; fi
done
