#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
import matplotlib.pyplot as plt

def create_structured_grids(internal_coords, internal_values, boundary_coords, boundary_values, domain):
    """
    Transforms internal and boundary point cloud data into two uniform, structured grids.

    This function implements a robust "fluid map" and neighbor check logic
    to correctly place boundary values into ghost cells adjacent to the internal fluid domain,
    handling complex geometries like internal obstacles.

    Args:
        internal_coords (np.ndarray): Array of shape (N, 2) for internal point coordinates (x, y).
        internal_values (np.ndarray): Array of shape (N, D) for values at internal points.
        boundary_coords (np.ndarray): Array of shape (M, 2) for boundary point coordinates (x, y).
        boundary_values (np.ndarray): Array of shape (M, D) for values at boundary points.
        domain (dict): Dictionary with domain metadata ('width', 'height', 'x_min', 'y_min', 'dx', 'dy', 'h_cells', 'v_cells').

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - internal_grid (np.ndarray): A (v_cells, h_cells, D) grid with internal values.
            - boundary_grid (np.ndarray): A (v_cells, h_cells, D) grid with boundary values correctly placed.
    """
    h_cells = domain['h_cells']
    v_cells = domain['v_cells']
    dx = domain['dx']
    dy = domain['dy']

    is_fluid_cell = np.zeros((v_cells, h_cells), dtype=bool)
    j_internal_base = np.floor((internal_coords[:, 0] - domain['x_min']) / dx).astype(int)
    i_internal_base = np.floor((internal_coords[:, 1] - domain['y_min']) / dy).astype(int)
    j_internal_base = np.clip(j_internal_base, 0, h_cells - 1)
    i_internal_base = np.clip(i_internal_base, 0, v_cells - 1)
    is_fluid_cell[i_internal_base, j_internal_base] = True

    padded_v_cells = v_cells + 2
    padded_h_cells = h_cells + 2
    padded_grid_shape = (padded_v_cells, padded_h_cells)
    internal_grid_padded = np.full(padded_grid_shape + (internal_values.shape[1],), np.nan, dtype=np.float32)
    boundary_grid_padded = np.full(padded_grid_shape + (boundary_values.shape[1],), np.nan, dtype=np.float32)
    internal_grid_padded[i_internal_base + 1, j_internal_base + 1] = internal_values

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
    boundary_grid_padded[i_pad_final, j_pad_final] = boundary_values
    
    internal_grid = internal_grid_padded[1:-1, 1:-1]
    boundary_grid = boundary_grid_padded[1:-1, 1:-1]

    return internal_grid, boundary_grid

# --- MAIN SCRIPT ---

if __name__ == "__main__":

    DATA_DIR = "data/fluid-openfoam/"
    file_path = DATA_DIR + "run_001.npz"
    X_CELLS = 128
    Y_CELLS = 64
    timestep_to_map = -1

    data = np.load(file_path, allow_pickle=True)
    internal_coords = data['internal_coordinates']
    internal_values = data['Solver-Mesh-2D-Internal'][timestep_to_map]
    boundary_coords = data['boundary_coordinates']
    boundary_values = data['Solver-Mesh-2D-Boundaries'][timestep_to_map]
    params_str = data['parameters'].item()
    params = json.loads(params_str)
    domain = params['domain']

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

    internal_magnitude = np.sqrt(internal_grid[:, :, 0]**2 + internal_grid[:, :, 1]**2)
    boundary_magnitude = np.sqrt(boundary_grid[:, :, 0]**2 + boundary_grid[:, :, 1]**2)

    global_min = np.nanmin([np.nanmin(internal_magnitude), np.nanmin(boundary_magnitude)])
    global_max = np.nanmax([np.nanmax(internal_magnitude), np.nanmax(boundary_magnitude)])

    plot_extent = [
        domain['x_min']-domain['dx'], domain['x_min'] + domain['width'] + domain['dx'],
        domain['y_min']-domain['dy'], domain['y_min'] + domain['height'] + domain['dy']
    ]

    plt.figure(figsize=(12, 6.5))
    ax = plt.gca()
    im1 = ax.imshow(internal_magnitude, cmap='coolwarm', origin='lower', aspect='equal', 
                    extent=plot_extent, vmin=global_min, vmax=global_max)
    im2 = ax.imshow(boundary_magnitude, cmap='coolwarm', origin='lower', aspect='equal', 
                    extent=plot_extent, vmin=global_min, vmax=global_max)
    ax.scatter(boundary_coords[:, 0], boundary_coords[:, 1], s=5, c='black', label='Boundary Coords')
    plt.colorbar(im1, ax=ax, label='Velocity Magnitude')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend(loc='lower right')
    plt.show()