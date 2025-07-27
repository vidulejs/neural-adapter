#!/bin/bash
set -e

echo "Starting datagen participant..."
cd "$(dirname "$0")"

# Get the number of epochs from config.json
epochs=$(python3 -c "import json; print(json.load(open('config.json'))['datagen']['epochs'])")

for (( i=0; i<epochs; i++ ))
do
   echo "--- Starting Datagen Epoch $i ---"
   python3 datagen.py $1 --epoch $i
done
