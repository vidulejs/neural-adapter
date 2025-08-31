# Data Generation

This directory contains the scripts for the data generation participant.

## How to Run

To run a simulation, you need to start both the data generator and a solver in two separate terminals.

The `run.sh` script takes the simulation dimension (1 or 2) as its first argument. You can also provide an optional second argument to specify the output directory for the generated data. If no output directory is provided, the data will be saved in the current directory (`datagen/`).

### 1D Simulation

**Terminal 1: Start Data Generator**
```bash
cd datagen

# Save data to the default directory (datagen/)
./run.sh 1

# --- OR ---

# Save data to a custom directory (e.g., a new folder called "my_1d_data")
# The path is relative to the datagen directory.
./run.sh 1 my_1d_data
```

**Terminal 2: Start Solver**
```bash
cd solver-dummy  # or solver-nutils
./run.sh 1
```

### 2D Simulation

**Terminal 1: Start Data Generator**
```bash
cd datagen

# Save data to the default directory (datagen/)
./run.sh 2

# --- OR ---

# Save data to a custom directory
./run.sh 2 my_2d_data
```

**Terminal 2: Start Solver**
```bash
cd solver-dummy  # or solver-nutils
./run.sh 2
```
