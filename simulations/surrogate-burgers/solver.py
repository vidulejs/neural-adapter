import torch
import numpy as np
import precice
import os
import sys
import json
import argparse
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

from neural_surrogate.model import CNN_RES
from neural_surrogate.utils import pad_with_ghost_cells
from neural_surrogate.config import HIDDEN_SIZE

# --- Model Configuration ---
MODEL_NAME = "neural_surrogate/models/CNN_RES_UNROLL_10.pth"
NUM_RES_BLOCKS = 4
KERNEL_SIZE = 5

def main(dim: int, config_file: str = "precice-config.xml"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Names ---
    read_data_name = "Data_1D"
    write_data_name = "SurrogateData"

    # --- Load pre-trained model ---
    model_path = os.path.join(project_root, MODEL_NAME)
    model = CNN_RES(
        hidden_channels=HIDDEN_SIZE,
        num_blocks=NUM_RES_BLOCKS,
        kernel_size=KERNEL_SIZE
    )
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        sys.exit(1)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Neural surrogate model loaded successfully.")

    # --- preCICE Setup ---
    with open(os.path.join(script_dir, "..", "python_participant", "config.json"), 'r') as f:
        config = json.load(f)["solver"]

    config_path = os.path.join(script_dir, "..", f"{dim}d", config_file)
    participant = precice.Participant("Surrogate", config_path, 0, 1)

    res = config[f"{dim}d_resolution"]
    domain_min = config[f"{dim}d_domain_min"]
    domain_max = config[f"{dim}d_domain_max"]

    domain_length = domain_max[0] - domain_min[0]
    shifted_domain_min = [domain_min[0] + domain_length, domain_min[1]]
    shifted_domain_max = [domain_max[0] + domain_length, domain_max[1]]

    boundary_mesh_name = "Surrogate-Boundaries"
    internal_mesh_name = "Surrogate-Internal"
    
    boundary_coords = np.array([[shifted_domain_min[0], shifted_domain_min[1]], [shifted_domain_max[0], shifted_domain_max[1]]])
    boundary_vertex_ids = participant.set_mesh_vertices(boundary_mesh_name, boundary_coords)

    internal_coords = np.array([np.linspace(shifted_domain_min[0], shifted_domain_max[0], res[0]), np.full(res[0], shifted_domain_min[1])]).T
    internal_vertex_ids = participant.set_mesh_vertices(internal_mesh_name, internal_coords)

    x_coords = np.linspace(domain_min[0], domain_max[0], res[0])
    initial_condition = -np.sin(2 * np.pi * 2 * x_coords / domain_length)
    # initial_condition *= np.linspace(10, 0, res[0])**2 / 100  # Damping towards the right boundary
    current_x = initial_condition

    participant.initialize()

    dt = participant.get_max_time_step_size()
    is_first_step = True
    total_solver_time = 0.0


    # --- Main Coupling Loop ---
    with torch.no_grad():
        while participant.is_coupling_ongoing():
            if is_first_step:
                # For the first step, use own boundaries for continuity
                bc_right_val = current_x[-1]
                bc_left_val = current_x[0]
                is_first_step = False
            else:
                # For subsequent steps, read from the other participant
                solver_boundary_values = participant.read_data(boundary_mesh_name, read_data_name, boundary_vertex_ids, dt)
                bc_right_val = solver_boundary_values[1] # First element of solver mesh
                bc_left_val = solver_boundary_values[0]  # Last element of solver mesh

            input_tensor = torch.from_numpy(current_x).float().unsqueeze(0).to(device)

            bc_left_tensor = torch.tensor([[bc_left_val]], device=device, dtype=torch.float)
            bc_right_tensor = torch.tensor([[bc_right_val]], device=device, dtype=torch.float)

            padded_input = torch.cat([bc_left_tensor, input_tensor, bc_right_tensor], dim=1)

            start_time = time.perf_counter()
            pred_tensor = model(padded_input)
            end_time = time.perf_counter()
            total_solver_time += (end_time - start_time)


            current_x = pred_tensor.squeeze().cpu().numpy()

            # --- 4. Write data to preCICE ---
            surrogate_boundary_values = np.array([current_x[0], current_x[-1]])
            participant.write_data(boundary_mesh_name, write_data_name, boundary_vertex_ids, surrogate_boundary_values)
            participant.write_data(internal_mesh_name, write_data_name, internal_vertex_ids, current_x)

            # --- 5. Advance coupling ---
            participant.advance(dt)

    participant.finalize()
    print("preCICE coupling finished.")
    print(f"Total solver computation time: {total_solver_time:.4f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dim", type=int, choices=[1, 2], help="Dimension of the simulation")
    parser.add_argument('--config_file', type=str, default="precice-config.xml")
    args_cli = parser.parse_args()

    main(dim=args_cli.dim, config_file=args_cli.config_file)
