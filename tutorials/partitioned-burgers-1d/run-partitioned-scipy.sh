#!/usr/bin/env bash
set -e -u

rm precice-run -rf

python3 generate_ic.py --epoch 10

cd dirichlet-scipy; pwd; ./run.sh &
cd ../neumann-scipy; pwd; ./run.sh && cd ..

python3 visualize_partitioned_domain.py
