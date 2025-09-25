"""
This script solves the 1D viscous Burgers' equation for a partitioned domain problem.
The two participants:
- 'Dirichlet': Solves the left half of the domain.
- 'Neumann':   Solves the right half of the domain.

# |<------- Dirichlet ------->|<------- Neumann ------->|
# | u(0)=0                    |                   u(n)=0|
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
import precice
import json
import os
import argparse
import time

def lax_friedrichs_flux(u_left, u_right): # Flux at cell interface between i-1/2 and i+1/2
    return 0.5 * (0.5 * u_left**2 + 0.5 * u_right**2) - 0.5 * (u_right - u_left)

def burgers_rhs(t, u, dx, C, bc_left, bc_right):
    # Ghost cells for BCs
    u_padded = np.empty(len(u) + 2)
    u_padded[0] = bc_left
    u_padded[-1] = bc_right
    u_padded[1:-1] = u

    flux = np.empty(len(u) + 1)
    for i in range(len(flux)):
        flux[i] = lax_friedrichs_flux(u_padded[i], u_padded[i+1])

    # Numerical viscosity
    viscosity = C * (u_padded[2:] - 2 * u_padded[1:-1] + u_padded[:-2]) / dx**2
    return -(flux[1:] - flux[:-1]) / dx + viscosity

# def burgers_jacobian(t, u, dx, C, bc_left, bc_right):
#     n = len(u)
#     # Derivatives of the Lax-Friedrichs flux
#     # dF/du_L = 0.5 * u_L + 0.5
#     # dF/du_R = 0.5 * u_R - 0.5
#     
#     # Contribution of flux_{i+1/2} to the diagonal at i
#     d_flux_di_plus = -(0.5 * u + 0.5) / dx
#     # Contribution of flux_{i-1/2} to the diagonal at i
#     d_flux_di_minus = (0.5 * u - 0.5) / dx
#     
#     # Off-diagonals
#     d_flux_d_up1 = -(0.5 * u[1:] - 0.5) / dx
#     d_flux_d_um1 = (0.5 * u[:-1] + 0.5) / dx
#
#     # --- Viscous Part ---
#     d_visc_di = -2 * C / dx**2
#     d_visc_off = C / dx**2
#
#     main_diag = d_flux_di_plus + d_flux_di_minus + d_visc_di
#     upper_diag = d_flux_d_up1 + d_visc_off
#     lower_diag = d_flux_d_um1 + d_visc_off
#
#     # Construct Jacobian
#     jac = diags([main_diag, upper_diag, lower_diag], [0, 1, -1], shape=(n, n), format='csc')
#
#     return jac

def main(participant_name: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    case_dir = os.path.abspath(os.path.join(script_dir, '..'))

    config_path = os.path.join(case_dir, "precice-config.xml")
    participant = precice.Participant(participant_name, config_path, 0, 1)

    # Read initial condition
    with open(os.path.join(case_dir, "ic_params.json"), 'r') as f:
        domain_config = json.load(f)["domain"]

    nelems_total = domain_config["nelems_total"]
    nelems_local = nelems_total // 2
    full_domain_min = domain_config["full_domain_min"]
    full_domain_max = domain_config["full_domain_max"]
    dx = (full_domain_max - full_domain_min) / nelems_total

    # Set domain and preCICE setup
    if participant_name == "Dirichlet":
        mesh_name = "Dirichlet-Mesh"
        read_data_name = "Velocity"
        write_data_name = "Flux"
        local_domain_min = full_domain_min
        local_domain_max = full_domain_min + nelems_local * dx
        coupling_point = [[local_domain_max, 0.0]]
    elif participant_name == "Neumann":
        mesh_name = "Neumann-Mesh"
        read_data_name = "Flux"
        write_data_name = "Velocity"
        local_domain_min = full_domain_min + nelems_local * dx
        local_domain_max = full_domain_max
        coupling_point = [[local_domain_min, 0.0]]
    else:
        raise ValueError(f"Unknown participant name: {participant_name}")

    vertex_id = participant.set_mesh_vertices(mesh_name, coupling_point)

    ic_data = np.load(os.path.join(case_dir, "initial_condition.npz"))
    full_ic = ic_data['initial_condition']
    if participant_name == "Dirichlet":
        u = full_ic[:nelems_local]
    else:
        u = full_ic[nelems_local:]

    solution_history = {0.0: u.copy()}

    participant.initialize()


    dt = participant.get_max_time_step_size()
    t = 0.0
    C_viscosity = 1e-12

    while participant.is_coupling_ongoing():
        if participant.requires_writing_checkpoint():
            saved_u = u.copy()

        # --- Read data and set BCs --- 
        if t > 0:
            if participant_name == "Dirichlet":
                bc_right = participant.read_data(mesh_name, read_data_name, vertex_id, dt)[0]
                print(f"[Dirichlet @ t={t:.2f}] Read bc_right = {bc_right:.4f}")
            else: # Neumann
                du_dx_bc = participant.read_data(mesh_name, read_data_name, vertex_id, dt)[0]
                print(f"[Neumann   @ t={t:.2f}] Read du/dx = {du_dx_bc:.4f}")
        else: # First timestep
            if participant_name == "Dirichlet":
                bc_right = full_ic[nelems_local]
                print(f"[Dirichlet @ t=0.00] Using initial bc_right = {bc_right:.4f}")
            else: # Neumann
                du_dx_bc = (full_ic[nelems_local] - full_ic[nelems_local - 1]) / dx
                print(f"[Neumann   @ t=0.00] Using initial du/dx = {du_dx_bc:.4f}")

        if participant_name == "Dirichlet":
            bc_left = 0
        else: # Neumann
            bc_left = u[0] - dx * du_dx_bc
            bc_right = 0

        # --- Solve one timestep ---
        t_end = t + dt
        solver_args = (dx, C_viscosity, bc_left, bc_right)
        sol = solve_ivp(burgers_rhs, (t, t_end), u, args=solver_args, method='BDF', t_eval=[t_end])

        u = sol.y[:, -1]
        t = sol.t[-1]

        solution_history[t] = u.copy()

        # --- Write coupling data ---
        if participant.is_coupling_ongoing():
            if participant_name == "Dirichlet":
                du_dx = (u[-1] - u[-2]) / dx
                participant.write_data(mesh_name, write_data_name, vertex_id, [du_dx])
            else: # Neumann
                participant.write_data(mesh_name, write_data_name, vertex_id, [u[0]])

        participant.advance(dt)

        if participant.requires_reading_checkpoint():
            u = saved_u.copy()

    # Finalize and save data to npz array
    participant.finalize()

    run_dir = os.getcwd()
    output_filename = os.path.join(run_dir, f"{participant_name.lower()}.npz")

    cell_centers_x = np.linspace(local_domain_min + dx/2, local_domain_max - dx/2, nelems_local)
    internal_coords = np.array([cell_centers_x, np.zeros(nelems_local)]).T

    sorted_times = sorted(solution_history.keys())
    final_solution = np.array([solution_history[time] for time in sorted_times])

    np.savez(
        output_filename,
        internal_coordinates=internal_coords,
        **{"Solver-Mesh-1D-Internal": final_solution}
    )
    print(f"[{participant_name}] Results saved to {output_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("participant", help="Name of the participant", choices=['Dirichlet', 'Neumann'])
    args = parser.parse_args()
    main(args.participant)