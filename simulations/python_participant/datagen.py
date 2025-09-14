import numpy as np
import precice
import os
import sys
import json
import argparse

class DataGenerator:
    def __init__(self, args):
        self.dim = args.dim
        self.epoch = args.epoch
        self.config_file = args.config
        self.output_path_base = args.output_path
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(self.script_dir, "config.json"), 'r') as f:
            self.config = json.load(f)

        config_path = os.path.abspath(os.path.join(self.script_dir, "..", f"{self.dim}d", self.config_file))
        self.participant = precice.Participant("DataGenerator", config_path, 0, 1)

        self.setup_mesh()
        self.current_time = 0.0

    def setup_mesh(self):
        if self.dim == 1:
            self.mesh_internal_name = "Solver-Mesh-1D-Internal"
            self.mesh_boundaries_name = "Solver-Mesh-1D-Boundaries"
            self.data_name = "Data_1D"
        elif self.dim == 2:
            self.mesh_internal_name = "Solver-Mesh-2D-Internal"
            self.mesh_boundaries_name = "Solver-Mesh-2D-Boundaries"
            self.data_name = "Data_2D"


    def run(self):
        if self.dim == 1:
            solver_config = self.config["solver"]
            domain_min = solver_config["1d_domain_min"]
            domain_max = solver_config["1d_domain_max"]
            epsilon = 1e-5
            bounding_box = [
                domain_min[0] - epsilon, domain_max[0] + epsilon,
                domain_min[1] - epsilon, domain_max[1] + epsilon
            ]
            self.participant.set_mesh_access_region(self.mesh_internal_name, bounding_box)
            self.participant.set_mesh_access_region(self.mesh_boundaries_name, bounding_box)

        self.participant.initialize()

        # Get mesh info from solver AFTER initialize
        self.internal_vertex_ids, self.internal_coords = self.participant.get_mesh_vertex_ids_and_coordinates(self.mesh_internal_name)
        self.boundary_vertex_ids, self.boundary_coords = self.participant.get_mesh_vertex_ids_and_coordinates(self.mesh_boundaries_name)

        self.data = {
            self.mesh_internal_name: [],
            self.mesh_boundaries_name: [],
            "internal_coordinates": self.internal_coords,
            "boundary_coordinates": self.boundary_coords
        }

        # Read initial data
        self.read_data()

        dt = self.participant.get_max_time_step_size()

        while self.participant.is_coupling_ongoing():
            try:
                self.participant.advance(dt)
                self.read_data()
                self.current_time += dt
            except precice.Error:
                print("Solver disconnected, finalizing datagen.")
                break

        self.participant.finalize()
        self.save_data()

    def read_data(self):
        print(f"DATAGEN: Reading data at simulation time {self.current_time:.4f}...")
        internal_read_values = self.participant.read_data(self.mesh_internal_name, self.data_name, self.internal_vertex_ids, 0.0)
        self.data[self.mesh_internal_name].append(internal_read_values)
        boundary_read_values = self.participant.read_data(self.mesh_boundaries_name, self.data_name, self.boundary_vertex_ids, 0.0)
        self.data[self.mesh_boundaries_name].append(boundary_read_values)

    def save_data(self):
        output_file = f"burgers_data_epoch_{self.epoch}.npz"
        output_path = os.path.join(self.output_path_base, output_file)
        os.makedirs(self.output_path_base, exist_ok=True)

        self.data[self.mesh_internal_name] = np.array(self.data[self.mesh_internal_name])
        self.data[self.mesh_boundaries_name] = np.array(self.data[self.mesh_boundaries_name])

        np.savez(output_path, **self.data)
        print(f"Data for epoch {self.epoch} saved to {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    default_output_dir = os.path.join(project_root, 'data', 'solver-nutils')

    parser = argparse.ArgumentParser()
    parser.add_argument("dim", type=int, choices=[1, 2], help="Dimension of the simulation")
    parser.add_argument("--epoch", type=int, default=0, help="Current epoch number")
    parser.add_argument("--config", type=str, default="precice-config.xml", help="preCICE configuration file")
    parser.add_argument("--output-path", type=str, default=default_output_dir, help="Directory to save the generated data")
    args = parser.parse_args()

    datagen = DataGenerator(args)
    datagen.run()