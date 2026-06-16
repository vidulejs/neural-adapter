#!/usr/bin/env bash
set -e -u

rm precice-run -rf

python3 utils/generate_ic.py --epoch ${1:-0}

cd dirichlet-scipy; pwd; ./run.sh &
cd ../neumann-surrogate; pwd; ./run.sh && cd ..

# full domain reference solution
cd solver-scipy; python3 solver.py None; cd ..


python3 utils/visualize_partitioned_domain.py --neumann neumann-surrogate/surrogate.npz