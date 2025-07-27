#!/bin/bash

# Get the number of epochs from config.json
cd "$(dirname "$0")"
epochs=$(python3 -c "import json; print(json.load(open('../datagen/config.json'))['datagen']['epochs'])")

for (( i=0; i<epochs; i++ ))
do
   echo "--- Starting Solver Epoch $i ---"
   python3 solver.py $1
done
