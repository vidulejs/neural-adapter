#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import json
import os


# In[3]:


# --- Configuration ---
file_path = "data/fluid-openfoam/run_095.npz"

internal_mesh_key = "Solver-Mesh-2D-Internal"
boundary_mesh_key = "Solver-Mesh-2D-Boundaries"
internal_coords_key = "internal_coordinates"
boundary_coords_key = "boundary_coordinates"


timestep = 0
# -------------------

data = np.load(file_path, allow_pickle=True)

params_str = data['parameters'].item()
params = json.loads(params_str)

print("--- Parameters ---")
print(json.dumps(params, indent=2))
print("--------------------")



if internal_mesh_key not in data:
    print(f"Error: Key '{internal_mesh_key}' not found in the data file.")
    print(f"Available keys: {list(data.keys())}")

velocity_data = data[internal_mesh_key]

if timestep >= velocity_data.shape[0] or timestep < -velocity_data.shape[0]:
    print(f"Error: Timestep {timestep} is out of bounds.")
    print(f"Available timesteps: 0 to {velocity_data.shape[0] - 1}")

print(f"Timesteps {velocity_data.shape[0]}")

selected_velocity = velocity_data[timestep, :, :]

magnitude = np.sqrt(selected_velocity[:, 0]**2 + selected_velocity[:, 1]**2)
internal_coords = data[internal_coords_key]

# Get the boundary data
boundary_coords = data[boundary_coords_key]
boundary_values = data[boundary_mesh_key][timestep, :, :]
boundary_magnitudes = np.sqrt(boundary_values[:, 0]**2 + boundary_values[:, 1]**2)


plt.figure(figsize=(12, 6))
sc = plt.scatter(internal_coords[:, 0], internal_coords[:, 1], c=magnitude, cmap='coolwarm', marker='.', s=10, label="Internal Points")
plt.scatter(boundary_coords[:, 0], boundary_coords[:, 1], c=boundary_magnitudes, cmap='coolwarm', marker='x', s=20, label="Boundary Points")

plt.colorbar(sc, label='Velocity Magnitude')
plt.title(f"Combined Mesh Visualization (Timestep {timestep})")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()