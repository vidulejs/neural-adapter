# Neural Adapter

A framework for generating simulation data and training neural network surrogates for preCICE-coupled solvers.

---

### Project Structure

-   `./simulations/`: Contains all components for generating and running coupled simulations.
    -   `fluid-openfoam/`: The OpenFOAM participant, case templates, and parameter sets.
    -   `python_participant/`: The Python-based data recorder participant.
-   `./data/`: The output directory for all generated simulation data. This directory is git-ignored.
-   `./neural_surrogate/`: Contains the PyTorch machine learning models, dataset handlers, and training scripts.

---

### Core Workflow
