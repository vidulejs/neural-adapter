import numpy as np
import matplotlib.pyplot as plt
import os

TIMESTEP_TO_PLOT = 100 #eg. 0, 1, ..., 100, ... ,-1

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIRICHLET_DATA_PATH = os.path.join(CASE_DIR, "dirichlet-scipy", "dirichlet.npz")
NEUMANN_DATA_PATH = os.path.join(CASE_DIR, "neumann-scipy", "neumann.npz")

print(f"Loading data from {DIRICHLET_DATA_PATH}")
data_d = np.load(DIRICHLET_DATA_PATH)
coords_d = data_d['internal_coordinates']
solution_d = data_d['Solver-Mesh-1D-Internal']

print(f"Loading data from {NEUMANN_DATA_PATH}")
data_n = np.load(NEUMANN_DATA_PATH)
coords_n = data_n['internal_coordinates']
solution_n = data_n['Solver-Mesh-1D-Internal']

full_coords = np.concatenate((coords_d[:, 0], coords_n[:, 0]))
full_solution_history = np.concatenate((solution_d, solution_n), axis=1)

print(f"Full domain shape: {full_solution_history.shape}")

# --- plot single timestep ---
plt.figure(figsize=(10, 5))
plt.plot(full_coords, full_solution_history[TIMESTEP_TO_PLOT, :], marker='.', linestyle='-')
plt.title(f'Solution at Timestep {TIMESTEP_TO_PLOT}')
plt.xlabel('Spatial Coordinate (x)')
plt.ylabel('Solution Value (u)')
plt.grid(True)
plt.savefig(os.path.join(CASE_DIR, f'full_domain_timestep_{TIMESTEP_TO_PLOT}.png'))
print(f"Saved plot to full_domain_timestep_{TIMESTEP_TO_PLOT}.png")

# --- plot time evolution ---
plt.figure(figsize=(10, 6))
plt.imshow(full_solution_history.T, aspect='auto', cmap='viridis', origin='lower',
           extent=[0, full_solution_history.shape[0], full_coords.min(), full_coords.max()])
plt.colorbar(label='Solution Value (u)')
plt.title('Time Evolution of Partitioned Burgers eq.')
plt.xlabel('Timestep')
plt.ylabel('Spatial Coordinate (x)')
plt.tight_layout()
plt.savefig(os.path.join(CASE_DIR, 'full_domain_evolution.png'))
print("Saved plot to full_domain_evolution.png")

plt.show()
