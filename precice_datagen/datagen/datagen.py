import numpy as np
import precice
import os
import sys
import json
import argparse

class DataGenerator:
    def __init__(self, dim, epoch, config_file="precice-config.xml"):
        self.dim = dim
        self.epoch = epoch
        self.config_file = config_file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        with open(os.path.join(script_dir, "config.json"), 'r') as f:
            self.config = json.load(f)["datagen"]

        config_path = os.path.abspath(os.path.join(script_dir, "..", f"{self.dim}d", self.config_file))
        self.participant = precice.Participant("DataGenerator", config_path, 0, 1)

        self.setup_mesh()

    def setup_mesh(self):
        if self.dim == 1:
            self.mesh_internal_name = "DataGenerator-Mesh-1D-Internal"
            self.mesh_boundaries_name = "DataGenerator-Mesh-1D-Boundaries"
            self.data_name = "Data_1D"
            self.ic_name = "Initial_Condition_1D"
            res = self.config["1d_resolution"]
            domain_min = self.config["domain_min"]
            domain_max = self.config["domain_max"]

            self.internal_coords = np.array([np.linspace(domain_min[0], domain_max[0], res[0]), np.full(res[0], domain_min[1])]).T
            self.internal_vertex_ids = self.participant.set_mesh_vertices(self.mesh_internal_name, self.internal_coords)

            self.boundary_coords = np.array([[domain_min[0], domain_min[1]], [domain_max[0], domain_min[1]]])
            self.boundary_vertex_ids = self.participant.set_mesh_vertices(self.mesh_boundaries_name, self.boundary_coords)

    def _generate_initial_condition(self, epoch):
        ic_config = self.config["initial_conditions"]
        np.random.seed(epoch)
        
        x_coords = self.internal_coords[:, 0]
        ic_values = np.zeros(len(self.internal_vertex_ids))

        if ic_config["type"] == "sinusoidal":
            num_modes = ic_config.get("num_modes", 1)
            for _ in range(num_modes):
                # Generate each mode with a random amplitude to create varied internal scales
                amp = np.random.uniform(0.1, 2)
                k = np.random.randint(ic_config["wavenumber_range"][0], ic_config["wavenumber_range"][1] + 1)
                phase_shift = np.random.uniform(0, 2 * np.pi)
                ic_values += amp * np.sin(2 * np.pi * k * x_coords + phase_shift)
        
            
        return ic_values

    def run(self):
        initial_condition = self._generate_initial_condition(self.epoch)
        if self.participant.requires_initial_data():
            self.participant.write_data(self.mesh_internal_name, self.ic_name, self.internal_vertex_ids, initial_condition)

        self.participant.initialize()
        
        dt = self.participant.get_max_time_step_size()

        boundary_indices = [np.where((self.internal_coords == b).all(axis=1))[0][0] for b in self.boundary_coords]
        initial_boundary_condition = initial_condition[boundary_indices]

        self.data = {
            self.mesh_internal_name: [initial_condition],
            self.mesh_boundaries_name: [initial_boundary_condition]
        }

        while self.participant.is_coupling_ongoing():
            self.participant.advance(dt)
            self.read_data()

        self.participant.finalize()
        self.save_data()

    def read_data(self):
        if self.dim == 1:
            internal_read_values = self.participant.read_data(self.mesh_internal_name, self.data_name, self.internal_vertex_ids, 0.0)
            self.data[self.mesh_internal_name].append(internal_read_values)
            boundary_read_values = self.participant.read_data(self.mesh_boundaries_name, self.data_name, self.boundary_vertex_ids, 0.0)
            self.data[self.mesh_boundaries_name].append(boundary_read_values)

    def save_data(self):
        if self.dim == 1:
            output_file = f"burgers_data_epoch_{self.epoch}.npz"
            np.savez(output_file, **self.data)
            print(f"Data for epoch {self.epoch} saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dim", type=int, choices=[1, 2], help="Dimension of the simulation")
    parser.add_argument("--epoch", type=int, default=0, help="Current epoch number")
    parser.add_argument("--config", type=str, default="precice-config.xml", help="preCICE configuration file")
    args = parser.parse_args()

    datagen = DataGenerator(args.dim, args.epoch, args.config)
    datagen.run()