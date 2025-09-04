#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import matplotlib.pyplot as plt

def create_structured_grids(internal_coords, internal_values, boundary_coords, boundary_values, domain):
    """
    Transforms internal and boundary point cloud data into uniform, structured grids.
    This function handles both single timesteps and full time series.

    Args:
        internal_coords (np.ndarray): Shape (N, 2) for internal point coordinates.
        internal_values (np.ndarray): Shape (N, D) OR (T, N, D) for values at internal points.
        boundary_coords (np.ndarray): Shape (M, 2) for boundary point coordinates.
        boundary_values (np.ndarray): Shape (M, D) OR (T, M, D) for values at boundary points.
        domain (dict): Dictionary with domain and grid metadata.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple of grids.
            - internal_grid: Shape (v_cells, h_cells, D) OR (T, v_cells, h_cells, D)
            - boundary_grid: Shape (v_cells, h_cells, D) OR (T, v_cells, h_cells, D)
    """
    # Check if the input is a time series (e.g., shape (T, N, D)) or a single frame (N, D)
    is_timeseries = internal_values.ndim == 3
    if not is_timeseries:
        # If single frame, add a temporary time axis for consistent processing
        internal_values = internal_values[np.newaxis, ...]
        boundary_values = boundary_values[np.newaxis, ...]

    num_timesteps = internal_values.shape[0]
    num_features = internal_values.shape[2]

    h_cells, v_cells = domain['h_cells'], domain['v_cells']
    dx, dy = domain['dx'], domain['dy']

    is_fluid_cell = np.zeros((v_cells, h_cells), dtype=bool)
    j_internal_base = np.clip(np.floor((internal_coords[:, 0] - domain['x_min']) / dx).astype(int), 0, h_cells - 1)
    i_internal_base = np.clip(np.floor((internal_coords[:, 1] - domain['y_min']) / dy).astype(int), 0, v_cells - 1)
    is_fluid_cell[i_internal_base, j_internal_base] = True

    padded_shape = (num_timesteps, v_cells + 2, h_cells + 2, num_features)
    internal_grid_padded = np.full(padded_shape, np.nan, dtype=np.float32)
    boundary_grid_padded = np.full(padded_shape, np.nan, dtype=np.float32)

    # construct internal grid
    internal_grid_padded[:, i_internal_base + 1, j_internal_base + 1] = internal_values

    # Calculate final boundary indices
    normalized_x = (boundary_coords[:, 0] - domain['x_min']) / dx
    is_vertical_face = np.isclose(np.round(normalized_x), normalized_x, atol=1e-5)
    is_horizontal_face = ~is_vertical_face
    i_pad_final = np.zeros(len(boundary_coords), dtype=int)
    j_pad_final = np.zeros(len(boundary_coords), dtype=int)
    v_indices = np.where(is_vertical_face)[0]
    coords_v = boundary_coords[v_indices]
    i_base_v = np.clip(np.floor((coords_v[:, 1] - domain['y_min']) / dy).astype(int), 0, v_cells - 1)
    j_right_v = np.clip(np.round((coords_v[:, 0] - domain['x_min']) / dx).astype(int), 0, h_cells - 1)
    j_left_v = np.clip(j_right_v - 1, 0, h_cells - 1)
    fluid_is_on_left = is_fluid_cell[i_base_v, j_left_v]
    j_ghost_v = np.where(fluid_is_on_left, j_right_v + 1, j_left_v + 1)
    i_pad_final[v_indices] = i_base_v + 1
    j_pad_final[v_indices] = j_ghost_v
    h_indices = np.where(is_horizontal_face)[0]
    coords_h = boundary_coords[h_indices]
    j_base_h = np.clip(np.floor((coords_h[:, 0] - domain['x_min']) / dx).astype(int), 0, h_cells - 1)
    i_above_h = np.clip(np.round((coords_h[:, 1] - domain['y_min']) / dy).astype(int), 0, v_cells - 1)
    i_below_h = np.clip(i_above_h - 1, 0, v_cells - 1)
    fluid_is_below = is_fluid_cell[i_below_h, j_base_h]
    i_ghost_h = np.where(fluid_is_below, i_above_h + 1, i_below_h + 1)
    j_pad_final[h_indices] = j_base_h + 1
    i_pad_final[h_indices] = i_ghost_h
    
    # construct boundary grid
    boundary_grid_padded[:, i_pad_final, j_pad_final] = boundary_values
    
    # remove padding before returning
    internal_grid = internal_grid_padded[:, 1:-1, 1:-1, :]
    boundary_grid = boundary_grid_padded[:, 1:-1, 1:-1, :]

    if not is_timeseries:
        # If not timeseries, remove the time axis before returning
        internal_grid = np.squeeze(internal_grid, axis=0)
        boundary_grid = np.squeeze(boundary_grid, axis=0)

    return internal_grid, boundary_grid

# --- MAIN SCRIPT ---

if __name__ == "__main__":

    DATA_DIR = "data/fluid-openfoam/"
    file_path = DATA_DIR + "run_001.npz"
    X_CELLS = 128
    Y_CELLS = 64
    timestep_to_map = -1

    # --- Data Loading ---
    data = np.load(file_path, allow_pickle=True)
    internal_coords = data['internal_coordinates']
    boundary_coords = data['boundary_coordinates']
    params_str = data['parameters'].item()
    params = json.loads(params_str)
    domain = params['domain']

    internal_values = data['Solver-Mesh-2D-Internal']
    boundary_values = data['Solver-Mesh-2D-Boundaries']
    
    # --- Domain Setup ---
    domain['h_cells'] = X_CELLS
    domain['v_cells'] = Y_CELLS
    domain['dx'] = domain['width'] / domain['h_cells']
    domain['dy'] = domain['height'] / domain['v_cells']

    internal_grid, boundary_grid = create_structured_grids(
        internal_coords,
        internal_values,
        boundary_coords,
        boundary_values,
        domain
    )

    internal_grid_slice = internal_grid[timestep_to_map]
    boundary_grid_slice = boundary_grid[timestep_to_map]

    # --- Visualization ---
    internal_magnitude_slice = np.sqrt(internal_grid_slice[:, :, 0]**2 + internal_grid_slice[:, :, 1]**2)
    boundary_magnitude_slice = np.sqrt(boundary_grid_slice[:, :, 0]**2 + boundary_grid_slice[:, :, 1]**2)

    global_min = np.nanmin([np.nanmin(internal_magnitude_slice), np.nanmin(boundary_magnitude_slice)])
    global_max = np.nanmax([np.nanmax(internal_magnitude_slice), np.nanmax(boundary_magnitude_slice)])

    plot_extent = [
        domain['x_min']-domain['dx'], domain['x_min'] + domain['width'] + domain['dx'],
        domain['y_min']-domain['dy'], domain['y_min'] + domain['height'] + domain['dy']
    ]

    plt.figure(figsize=(12, 6.5))
    ax = plt.gca()
    im1 = ax.imshow(internal_magnitude_slice, cmap='coolwarm', origin='lower', aspect='equal', 
                    extent=plot_extent, vmin=global_min, vmax=global_max)
    im2 = ax.imshow(boundary_magnitude_slice, cmap='coolwarm', origin='lower', aspect='equal', 
                    extent=plot_extent, vmin=global_min, vmax=global_max)
    ax.scatter(boundary_coords[:, 0], boundary_coords[:, 1], s=5, c='black', label='Boundary Coords')
    plt.colorbar(im1, ax=ax, label='Velocity Magnitude')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='lower right')
    plt.show()