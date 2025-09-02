#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import json
import os


# In[3]:


# --- Configuration ---
file_path = "/home/dan/neural-adapter/precice_datagen/simulation_runs/run_002/flow_data_epoch_0.npz"

timestep_to_plot = -1

internal_mesh_key = "Solver-Mesh-2D-Internal"
boundary_mesh_key = "Solver-Mesh-2D-Boundaries"
internal_coords_key = "internal_coordinates"
boundary_coords_key = "boundary_coordinates"

config_path = "/home/dan/neural-adapter/precice_datagen/datagen/config.json"

timestep = 1
# -------------------

if not os.path.exists(config_path):
    print(f"Error: Configuration file not found at '{config_path}'")

data = np.load(file_path)

if internal_mesh_key not in data:
    print(f"Error: Key '{internal_mesh_key}' not found in the data file.")
    print(f"Available keys: {list(data.keys())}")


velocity_data = data[internal_mesh_key]

if timestep >= velocity_data.shape[0] or timestep < -velocity_data.shape[0]:
    print(f"Error: Timestep {timestep} is out of bounds.")
    print(f"Available timesteps: 0 to {velocity_data.shape[0] - 1}")

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