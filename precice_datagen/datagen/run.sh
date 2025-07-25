#!/bin/bash
set -e

echo "Starting datagen participant..."
cd "$(dirname "$0")"
python datagen.py "$@"
