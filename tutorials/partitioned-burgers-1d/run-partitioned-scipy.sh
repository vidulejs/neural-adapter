#!/usr/bin/env bash
set -e -u

python3 generate_ic.py --epoch 0

cd dirichlet-scipy; pwd; ./run.sh &
cd ../neumann-scipy; pwd; ./run.sh && cd ..

python3 visualize_partitioned_domain.py