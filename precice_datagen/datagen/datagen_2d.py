import numpy as np
import precice
import os
import sys
import json
import argparse

class DataGenerator:
    def __init__(self, epoch, config_file="precice-config.xml"):
        self.dim = 2
        self.epoch = epoch
        self.config_file = config_file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        with open(os.path.join(script_dir, "config.json"), 'r') as f:
            self.config = json.load(f)["datagen"]

        config_path = os.path.abspath(os.path.join(script_dir, "..", f"{self.dim}d", self.config_file))
        self.participant = precice.Participant("DataGenerator", config_path, 0, 1)

        self.setup_mesh()

    def setup_mesh(self):
        self.mesh_internal_name = "DataGenerator-Mesh-2D-Internal"
        self.mesh_boundaries_name = "DataGenerator-Mesh-2D-Boundaries"
        self.data_name = "Data_2D"
        
        res = self.config["2d_resolution"]
        domain_min = self.config["domain_min"]
        domain_max = self.config["domain_max"]

        x = np.linspace(domain_min[0], domain_max[0], res[0])
        y = np.linspace(domain_min[1], domain_max[1], res[1])
        xx, yy = np.meshgrid(x, y)
        self.internal_coords = np.vstack((xx.ravel(), yy.ravel())).T
        self.internal_vertex_ids = self.participant.set_mesh_vertices(self.mesh_internal_name, self.internal_coords)

        # Define boundaries (e.g., 4 corners of a rectangle)
        self.boundary_coords = np.array([
            [domain_min[0], domain_min[1]],
            [domain_max[0], domain_min[1]],
            [domain_min[0], domain_max[1]],
            [domain_max[0], domain_max[1]]
        ])
        self.boundary_vertex_ids = self.participant.set_mesh_vertices(self.mesh_boundaries_name, self.boundary_coords)

    def run(self):
        self.participant.initialize()
        
        dt = self.participant.get_max_time_step_size()

        self.data = {
            self.mesh_internal_name: [],
            self.mesh_boundaries_name: []
        }

        while self.participant.is_coupling_ongoing():
            self.participant.advance(dt)
            self.read_data()

        self.participant.finalize()
        self.save_data()

    def read_data(self):
        internal_read_values = self.participant.read_data(self.mesh_internal_name, self.data_name, self.internal_vertex_ids, 0.0)
        self.data[self.mesh_internal_name].append(internal_read_values)
        boundary_read_values = self.participant.read_data(self.mesh_boundaries_name, self.data_name, self.boundary_vertex_ids, 0.0)
        self.data[self.mesh_boundaries_name].append(boundary_read_values)

    def save_data(self):
        output_file = f"flow_data_epoch_{self.epoch}.npz"
        output_path = os.path.join(args.output_path, output_file)
        os.makedirs(args.output_path, exist_ok=True)
        
        # Convert lists to numpy arrays for saving
        for key in self.data:
            self.data[key] = np.array(self.data[key])
            
        np.savez(output_path, **self.data)
        print(f"Data for epoch {self.epoch} saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="Current epoch number")
    parser.add_argument("--config", type=str, default="precice-config.xml", help="preCICE configuration file")
    parser.add_argument("--output-path", type=str, default=".", help="Directory to save the generated data")
    args = parser.parse_args()

    datagen = DataGenerator(args.epoch, args.config)
    datagen.run()