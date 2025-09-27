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

    if participant.requires_initial_data():
        if participant_name == "Dirichlet":
            du_dx = (u[-1] - u[-2]) / dx
            participant.write_data(mesh_name, write_data_name, vertex_id, [du_dx])
        else: # Neumann
            participant.write_data(mesh_name, write_data_name, vertex_id, [u[0]])

    participant.initialize()

    dt = participant.get_max_time_step_size()
    t = 0.0
    saved_t = 0.0
    C_viscosity = 1e-12

    # --- Serial Coupling Loop ---
    if participant_name == "Dirichlet":
        while participant.is_coupling_ongoing():
            if participant.requires_writing_checkpoint():
                # print(f"[Dirichlet] Writing checkpoint at t={t:.4f}")
                saved_u = u.copy()
                saved_t = t
            if participant.requires_reading_checkpoint():
                u = saved_u.copy()
                t = saved_t
                # print(f"[Dirichlet] Reading checkpoint at t={t:.4f}")

            bc_right = participant.read_data(mesh_name, read_data_name, vertex_id, dt)[0]
            bc_left = 0

            t_end = t + dt
            solver_args = (dx, C_viscosity, bc_left, bc_right)
            sol = solve_ivp(burgers_rhs, (t, t_end), u, args=solver_args, method='BDF', t_eval=[t_end])
            u = sol.y[:, -1]
            
            du_dx = (u[-1] - u[-2]) / dx
            participant.write_data(mesh_name, write_data_name, vertex_id, [du_dx])

            print(f"[{participant_name:9s}] t={t:6.4f} | u_int={u[-1]:8.4f} | du/dx_loc={(u[-1]-u[-2])/dx:8.4f}")
            
            t = sol.t[-1]
            solution_history[t] = u.copy()
            participant.advance(dt)

    else: # Neumann
        while participant.is_coupling_ongoing():
            if participant.requires_writing_checkpoint():
                # print(f"[Neumann] Writing checkpoint at t={t:.4f}")
                saved_u = u.copy()
                saved_t = t
            if participant.requires_reading_checkpoint():
                u = saved_u.copy()
                t = saved_t
                # print(f"[Neumann] Reading checkpoint at t={t:.4f}")
                            
            du_dx_bc = participant.read_data(mesh_name, read_data_name, vertex_id, dt)[0]
            
            bc_left = u[0] - dx * du_dx_bc
            bc_right = 0

            t_end = t + dt
            solver_args = (dx, C_viscosity, bc_left, bc_right)
            sol = solve_ivp(burgers_rhs, (t, t_end), u, args=solver_args, method='BDF', t_eval=[t_end])
            u = sol.y[:, -1]

            participant.write_data(mesh_name, write_data_name, vertex_id, [u[0]])
            print(f"[{participant_name:9s}] t={t:6.4f} | u_int={u[0]:8.4f} | du/dx_loc={(u[1]-u[0])/dx:8.4f}")

            t = sol.t[-1]
            solution_history[t] = u.copy()
            participant.advance(dt)

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