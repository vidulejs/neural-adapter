#!/bin/bash
set -e

echo "Starting datagen participant..."
cd "$(dirname "$0")"

# Get the number of epochs from config.json
epochs=$(python3 -c "import json; print(json.load(open('config.json'))['datagen']['epochs'])")

for (( i=0; i<epochs; i++ ))
do
   echo "--- Starting Datagen Epoch $i ---"
   timeout 60s python3 datagen.py $1 --epoch $i || if [ $? -eq 124 ]; then echo "Datagen for epoch $i timed out. Continuing to next epoch."; else exit $?; fi
done
