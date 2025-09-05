import yaml
import os
import numpy as np
import random
import argparse

def generate_parameter_sets(start_case_num, num_cases_to_generate):
    """
    Generates a specified number of .yml parameter files by randomly sampling
    from a defined parameter space, ensuring a uniform mesh for each case.
    """
    # --- Mesh ---
    TOTAL_H_CELLS = 128
    TOTAL_V_CELLS = 64
    
    # --- Parameter space ---
    widths = np.arange(6.0, 12.0, 0.5, dtype=float)
    heights = np.arange(2.0, 6.0, 0.5, dtype=float)
    domain_options = [{'width': w, 'height': h} for w in widths for h in heights]

    v_cells_obstacle_fractions = np.linspace(0.25, 0.75, num=20, dtype=float)
    min_h_frac = 4 / TOTAL_H_CELLS
    h_cells_obstacle_fractions = np.linspace(min_h_frac, 0.5, num=20, dtype=float)
    h_cells_pre_obstacle_fractions = np.linspace(0.2, 0.8, num=20, dtype=float)

    inlet_velocity_options = np.arange(0.1, 10, 0.1, dtype=float)

    # from 1e-5 to 1e-1
    kinematic_viscosity_options = np.logspace(-5, -1, num=50, base=10.0, dtype=float)
    print(kinematic_viscosity_options)


    # --- Generation Loop ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "param_sets")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving {num_cases_to_generate} random parameter files to: {output_dir}")
    print(f"Starting from case number: {start_case_num}")

    generated_configs = set()
    case_count = 0
    case_num = start_case_num

    max_attempts = num_cases_to_generate * 100 
    attempts = 0

    while case_count < num_cases_to_generate and attempts < max_attempts:
        domain = random.choice(domain_options)
        v_frac = random.choice(v_cells_obstacle_fractions)
        h_frac = random.choice(h_cells_obstacle_fractions)
        h_pre_frac = random.choice(h_cells_pre_obstacle_fractions)
        velocity = random.choice(inlet_velocity_options)
        kinematic_viscosity = random.choice(kinematic_viscosity_options)

        config_key = (domain['width'], domain['height'], v_frac, h_frac, h_pre_frac, round(velocity, 4))
        if config_key in generated_configs:
            attempts += 1
            continue
        
        generated_configs.add(config_key)

        DOMAIN_WIDTH = domain['width']
        DOMAIN_HEIGHT = domain['height']
        DELTA_X = DOMAIN_WIDTH / TOTAL_H_CELLS
        DELTA_Y = DOMAIN_HEIGHT / TOTAL_V_CELLS

        v_cells_obs = int(TOTAL_V_CELLS * v_frac)
        h_cells_obs = int(TOTAL_H_CELLS * h_frac)
        h_cells_pre = int(TOTAL_H_CELLS * h_pre_frac)

        obstacle_height = v_cells_obs * DELTA_Y
        obstacle_x_min = h_cells_pre * DELTA_X
        obstacle_x_max = obstacle_x_min + (h_cells_obs * DELTA_X)

        h_cells_post = TOTAL_H_CELLS - h_cells_pre - h_cells_obs
        v_cells_above = TOTAL_V_CELLS - v_cells_obs
        
        if h_cells_post <= 0 or v_cells_above <= 0:
            attempts += 1
            continue

        params = {
            'domain': {
                'x_min': 0.0, 'y_min': 0.0,
                'width': float(DOMAIN_WIDTH), 'height': float(DOMAIN_HEIGHT),
                'h_cells': TOTAL_H_CELLS, 'v_cells': TOTAL_V_CELLS
            },
            'obstacle': {
                'x_min': float(round(obstacle_x_min, 6)),
                'x_max': float(round(obstacle_x_max, 6)),
                'height': float(round(obstacle_height, 6)),
            },
            'inlet_velocity': [float(velocity), 0, 0],
            'kinematic_viscosity': float(kinematic_viscosity),
            'simulation_control': {
                'endTime': 5.0,
                'deltaT': 0.01,
                'writeInterval': 20,
            },
            'grading': {
                'h1_cells': h_cells_pre, 'h2_cells': h_cells_obs,
                'h3_cells': h_cells_post, 'v1_cells': v_cells_obs,
                'v2_cells': v_cells_above,
            }
        }

        file_name = f"case_{case_num:03d}.yml"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, 'w') as f:
            yaml.dump(params, f, default_flow_style=False, sort_keys=False)
        
        print(f"Generated {file_name}")
        case_num += 1
        case_count += 1
        attempts += 1
    
    if case_count < num_cases_to_generate:
        print(f"Warning: Could only generate {case_count} unique cases out of {num_cases_to_generate} requested.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate parameter sets for OpenFOAM cases.")
    parser.add_argument(
        '--start-num',
        type=int,
        default=1,
        help='The starting number for case files (e.g., 1, 51, 101).'
    )
    parser.add_argument(
        '--num-cases',
        type=int,
        default=100,
        help='The total number of new cases to generate.'
    )
    args = parser.parse_args()

    generate_parameter_sets(args.start_num, args.num_cases)
