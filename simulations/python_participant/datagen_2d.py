import numpy as np
import precice
import os
import sys
import json
import argparse
import yaml

class DataGenerator:
    def __init__(self, epoch, config_file="precice-config.xml", param_file=None):
        self.dim = 2
        self.epoch = epoch
        self.config_file = config_file
        
        if not param_file:
            raise ValueError("A parameter file (--params) is required.")
            
        with open(param_file, 'r') as f:
            self.params = yaml.safe_load(f)
        
        print("--- DATAGEN PARTICIPANT ---")
        print(f"Using parameters: {json.dumps(self.params, indent=2)}")
        print("---------------------------")

        config_path = os.path.abspath(self.config_file)
        self.participant = precice.Participant("DataGenerator", config_path, 0, 1)

        self.setup_mesh()
        self.current_time = 0.0

    def setup_mesh(self):
        self.solver_mesh_internal_name = "Solver-Mesh-2D-Internal"
        self.solver_mesh_boundaries_name = "Solver-Mesh-2D-Boundaries"
        self.data_name = "Data_2D"

    def run(self):
        # Define the region of interest from the loaded parameters
        domain = self.params['domain']
        epsilon = 1e-5 # Small padding to avoid floating point issues
        x_min = domain.get('x_min', 0.0) - epsilon
        y_min = domain.get('y_min', 0.0) - epsilon
        x_max = x_min + domain['width'] + (2 * epsilon)
        y_max = y_min + domain['height'] + (2 * epsilon)
        bounding_box = [x_min, x_max, y_min, y_max]
        
        self.participant.set_mesh_access_region(self.solver_mesh_internal_name, bounding_box)
        self.participant.set_mesh_access_region(self.solver_mesh_boundaries_name, bounding_box)

        self.participant.initialize()

        # Get mesh info from solver AFTER initialize
        self.internal_vertex_ids, self.internal_coords = self.participant.get_mesh_vertex_ids_and_coordinates(self.solver_mesh_internal_name)
        self.boundary_vertex_ids, self.boundary_coords = self.participant.get_mesh_vertex_ids_and_coordinates(self.solver_mesh_boundaries_name)
        
        dt = self.participant.get_max_time_step_size()

        self.data = {
            self.solver_mesh_internal_name: [],
            self.solver_mesh_boundaries_name: [],
            "internal_coordinates": self.internal_coords,
            "boundary_coordinates": self.boundary_coords
        }

        while self.participant.is_coupling_ongoing():
            self.participant.advance(dt)
            self.read_data()
            self.current_time += dt

        self.participant.finalize()
        self.save_data()

    def read_data(self):
        print(f"DATAGEN: Reading data at simulation time {self.current_time:.4f}...")
        internal_read_values = self.participant.read_data(self.solver_mesh_internal_name, self.data_name, self.internal_vertex_ids, 0.0)
        self.data[self.solver_mesh_internal_name].append(internal_read_values)
        boundary_read_values = self.participant.read_data(self.solver_mesh_boundaries_name, self.data_name, self.boundary_vertex_ids, 0.0)
        self.data[self.solver_mesh_boundaries_name].append(boundary_read_values)

    def save_data(self):
        output_path = os.path.abspath(args.output_path)
        # Ensure the directory for the output file exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert lists to numpy arrays for saving
        self.data[self.solver_mesh_internal_name] = np.array(self.data[self.solver_mesh_internal_name])
        self.data[self.solver_mesh_boundaries_name] = np.array(self.data[self.solver_mesh_boundaries_name])
        
        # Add the case parameters to the output file as a JSON string
        self.data['parameters'] = json.dumps(self.params)
            
        np.savez(output_path, **self.data)
        print(f"Data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="Current epoch number")
    parser.add_argument("--config", type=str, default="precice-config.xml", help="preCICE configuration file")
    parser.add_argument("--output-path", type=str, required=True, help="Full path to save the output .npz file.")
    parser.add_argument("--params", type=str, required=True, help="Path to the YAML parameter file for this run.")
    args = parser.parse_args()

    datagen = DataGenerator(args.epoch, args.config, args.params)
    datagen.run()
