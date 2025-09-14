import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
import precice
import json
import os
import argparse
import treelog as log
import itertools
import time

def _generate_initial_condition(x_coords, ic_config, epoch):
    np.random.seed(epoch)
    ic_values = np.zeros(len(x_coords))
    if ic_config["type"] == "sinusoidal":
        num_modes = ic_config.get("num_modes", 1)
        superpositions = np.random.randint(2, num_modes + 1)
        for _ in range(superpositions):
            amp = np.random.uniform(0.1, 2)
            k = np.random.randint(ic_config["wavenumber_range"][0], ic_config["wavenumber_range"][1] + 1)
            phase_shift = np.random.uniform(0, 2 * np.pi)
            ic_values += amp * np.sin(2 * np.pi * k * x_coords + phase_shift)
    return ic_values

def project_initial_condition(domain_min, domain_max, nelems, ic_config, epoch):
    # 1. Generate a high-resolution "truth" on a fine grid
    fine_res = nelems * 10
    fine_x = np.linspace(domain_min[0], domain_max[0], fine_res, endpoint=False)
    fine_u = _generate_initial_condition(fine_x, ic_config, epoch)

    # 2. Average the high-resolution truth over each coarse cell
    u_projected = np.zeros(nelems)
    for i in range(nelems):
        cell_start = i * 10
        cell_end = (i + 1) * 10
        u_projected[i] = np.mean(fine_u[cell_start:cell_end])
        
    return u_projected

def lax_friedrichs_flux(u_left, u_right):
    return 0.5 * (0.5 * u_left**2 + 0.5 * u_right**2) - 0.5 * (u_right - u_left)

def burgers_rhs(t, u, dx, C):
    u_padded = np.empty(len(u) + 2)
    # Periodic boundary conditions using ghost cells
    u_padded[0] = u[-1]
    u_padded[-1] = u[0]
    u_padded[1:-1] = u

    flux = np.empty(len(u) + 1)
    for i in range(len(flux)):
        flux[i] = lax_friedrichs_flux(u_padded[i], u_padded[i+1])

    # Add numerical viscosity for stability
    viscosity = C * (u_padded[2:] - 2 * u_padded[1:-1] + u_padded[:-2]) / dx**2

    return -(flux[1:] - flux[:-1]) / dx + viscosity

def burgers_jacobian(t, u, dx, C):
    n = len(u)
    # Derivatives of the Lax-Friedrichs flux
    # dF/du_L = 0.5 * u_L + 0.5
    # dF/du_R = 0.5 * u_R - 0.5
    
    # Main diagonal
    d_flux_di = -(0.5 * u + 0.5) / dx  # Contribution from flux_{i+1/2}
    d_flux_di += (0.5 * u - 0.5) / dx  # Contribution from flux_{i-1/2}
    
    # Off-diagonals
    d_flux_d_up1 = -(0.5 * u[1:] - 0.5) / dx
    d_flux_d_um1 = (0.5 * u[:-1] + 0.5) / dx

    # Viscosity contribution
    d_visc_di = -2 * C / dx**2
    d_visc_off = C / dx**2

    main_diag = d_flux_di + d_visc_di
    upper_diag = d_flux_d_up1 + d_visc_off
    lower_diag = d_flux_d_um1 + d_visc_off

    # Construct sparse Jacobian
    jac = diags([main_diag, upper_diag, lower_diag], [0, 1, -1], shape=(n, n), format='csc')
    
    # Handle periodic boundaries
    jac[0, -1] += (0.5 * u[-1] + 0.5) / dx + d_visc_off # Lower-left corner
    jac[-1, 0] += -(0.5 * u[0] - 0.5) / dx + d_visc_off # Upper-right corner

    return jac

def main(dim: int, epoch: int, config_file: str, coupling_mode: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "..", "python_participant", "config.json"), 'r') as f:
        config = json.load(f)["solver"]
    with open(os.path.join(script_dir, "ic_params.json"), 'r') as f:
        ic_config = json.load(f)["initial_conditions"]

    config_path = os.path.join(script_dir, "..", f"{dim}d", config_file)
    participant = precice.Participant("Solver", config_path, 0, 1)

    mesh_internal_name = f"Solver-Mesh-{dim}D-Internal"
    mesh_boundaries_name = f"Solver-Mesh-{dim}D-Boundaries"
    data_name = f"Data_{dim}D"

    res = config[f"{dim}d_resolution"]
    domain_min = config[f"{dim}d_domain_min"]
    domain_max = config[f"{dim}d_domain_max"]
    nelems = res[0]
    dx = (domain_max[0] - domain_min[0]) / nelems

    # Cell centers for finite volume method
    internal_coords_x = np.linspace(domain_min[0] + dx/2, domain_max[0] - dx/2, nelems)
    internal_coords = np.array([internal_coords_x, np.full(nelems, domain_min[1])]).T
    boundary_coords = np.array([[domain_min[0], domain_min[1]], [domain_max[0], domain_max[1]]])

    internal_vertex_ids = participant.set_mesh_vertices(mesh_internal_name, internal_coords)
    boundary_vertex_ids = participant.set_mesh_vertices(mesh_boundaries_name, boundary_coords)

    u = project_initial_condition(domain_min, domain_max, nelems, ic_config, epoch)
    
    if participant.requires_initial_data():
        boundary_data_values = np.array([u[0], u[-1]])
        participant.write_data(mesh_internal_name, data_name, internal_vertex_ids, u)
        participant.write_data(mesh_boundaries_name, data_name, boundary_vertex_ids, boundary_data_values)

    participant.initialize()

    t = 0.0
    total_solver_time = 0.0
    C_viscosity = 1e-12 # Artificial viscosity coefficient

    with log.iter.plain('timestep', itertools.count()) as steps:
        for _ in steps:
            if not participant.is_coupling_ongoing():
                break

            dt = participant.get_max_time_step_size()
            t_end = t + dt

            start_time = time.perf_counter()
            sol = solve_ivp(burgers_rhs, (t, t_end), u, args=(dx, C_viscosity), method='BDF', t_eval=[t_end], jac=burgers_jacobian)
            end_time = time.perf_counter()
            total_solver_time += (end_time - start_time)

            if sol.status != 0:
                print(f"Solver failed at t={t:.2f}: {sol.message}")
                break
            
            u = sol.y[:, -1]
            t = sol.t[-1]

            boundary_data_values = np.array([u[0], u[-1]])
            participant.write_data(mesh_internal_name, data_name, internal_vertex_ids, u)
            participant.write_data(mesh_boundaries_name, data_name, boundary_vertex_ids, boundary_data_values)

            participant.advance(dt)

    participant.finalize()
    print(f"Total solver computation time: {total_solver_time:.4f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dim", type=int, choices=[1], help="Dimension of the simulation (only 1D supported)")
    parser.add_argument("--epoch", type=int, default=0, help="Current epoch number")
    parser.add_argument('--config_file', type=str, default="precice-config.xml")
    parser.add_argument('--coupling-mode', type=str, default='datagen', choices=['datagen', 'coupled'])
    args_cli = parser.parse_args()
    main(dim=args_cli.dim, epoch=args_cli.epoch, config_file=args_cli.config_file, coupling_mode=args_cli.coupling_mode)
