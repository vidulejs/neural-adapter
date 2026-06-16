# Neural Adapter

 <!-- Let's relax framework to something less ambitious. like a dev repo -->
A framework for generating simulation data and training neural-network (NN)
surrogates for [preCICE](https://precice.org/)-coupled solvers.

This is the development code used in the master's thesis *"Coupling Neural
Surrogates with Traditional Solvers using preCICE"* (Dagis Daniels Vidulejs,
Technical University of Munich, 2025).

---

## Overview

Partitioned simulations build multi-physics problems by coupling specialised
single-physics solvers. This project explores replacing one such solver with a
neural surrogate inside a live, bidirectional preCICE coupling. The reference
case is a **boundary-aware, time-stepping NN surrogate for the 1D viscous
Burgers' equation**, coupled to a traditional Finite Volume Method (FVM) solver
through a Dirichlet–Neumann scheme on a non-overlapping interface.

The surrogate is a residual convolutional network (`CNN_RES`) that predicts the
next time-step solution from the current one. It is trained autoregressively
with backpropagation-through-time (BPTT) on data produced by the FVM (SciPy)
solver, and then used as a drop-in preCICE participant.

---

## Project Structure
- `neural_surrogate/` - PyTorch surrogate model code:
  - `model.py` - architecture definitions (`CNN_RES` residual CNN, MLP,
    ghost-cell padding).
  - `dataset.py` - dataset generation.
  - `config.py` - configuration file. Also training hyperparameters.
  - `train_burgers_bptt.ipynb` - **main training notebook for the 1D Burgers surrogate.**
  - `evaluate.py` - evaluation script.
  - `models/` - trained checkpoints (git-ignored).
- `tutorials/partitioned-burgers-1d/` - the 1D Burgers case (mirrors the preCICE tutorial):
  - `solver-scipy-fvolumes/` - the FVM solver, used both as a coupled participant and to
    generate monolithic training data.
  - `neumann-surrogate/` - the surrogate preCICE participant.
  - `generate-training-data.sh`, `generate_ic.py` - data-generation helpers.
- `interface/`, `precice_datagen/` - preCICE participants and run configurations for
  generating CFD (channel-transport tutorial) data.

---

## Setup

Tested with Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs the CPU build of PyTorch by default. For GPU
training, install a CUDA build of `torch` matching your hardware. Dependency
versions are very relaxed.

For CUDA support you can install the latest nightly PyTorch version, which
should work on Nvidia Blackwell GPUs as well.

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

Only `numpy`, `scipy` and `torch` are required for the core data generation and
training workflow below. `pyprecice` is only needed to run the actual coupled
simulations.

---

## Workflow (1D Burgers surrogate)

1. **Generate training data.** From `tutorials/partitioned-burgers-1d/`, run the
   monolithic FVM solver for many random initial conditions:

   ```bash
   cd tutorials/partitioned-burgers-1d
   ./generate-training-data.sh
   ```

   This calls `generate_ic.py --epoch <i>` followed by `solver.py None`
   (monolithic, no preCICE) for each run, writing
   `solver-scipy-fvolumes/data-training/burgers_data_epoch_*.npz`. That
   directory is the `DATA_DIR` used by `neural_surrogate/config.py`. Reduce
   `NUM_RUNS` in the script for a quick test.

2. **Train the surrogate.** Run `neural_surrogate/train_burgers_bptt.ipynb`. It
   trains a curriculum of models over `unroll_array` (e.g. `[1, 7]`) with BPTT,
   saving one checkpoint per unroll length to
   `neural_surrogate/models/CNN_RES_UNROLL_<N>.pth` together with a loss plot.
   To run it headless:

   ```bash
   cd neural_surrogate
   jupyter nbconvert --to notebook --execute train_burgers_bptt.ipynb
   ```

   Longer unroll lengths (e.g. 7) improve stability over long autoregressive
   rollouts.

3. **Evaluate.** The notebook's final cells perform an autoregressive rollout
   against the FVM ground truth and save `rollout_comparison.svg`.

4. **Deploy.** Copy the trained checkpoint into the surrogate participant
   (`tutorials/partitioned-burgers-1d/neumann-surrogate/`, or the corresponding
   preCICE tutorial) and set `MODEL_NAME` in its `config.py`.

