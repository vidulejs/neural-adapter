import numpy as np
import precice
import os
import sys
import json

class DataGenerator:
    def __init__(self, dim):
        self.dim = dim
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        with open(os.path.join(script_dir, "config.json"), 'r') as f:
            self.config = json.load(f)["datagen"]

        config_path = os.path.join(script_dir, "..", f"{self.dim}d", "precice-config.xml")
        self.participant = precice.Participant("DataGenerator", config_path, 0, 1)

        self.setup_mesh()
        self.participant.initialize()
        self.data = {}

    def setup_mesh(self):
        if self.dim == 1:
            self.mesh_name = "DataGenerator-Mesh-1D"
            self.data_name = "Data_1D"
            res = self.config["1d_resolution"]
            domain_min = self.config["domain_min"]
            domain_max = self.config["domain_max"]

            coords = np.linspace(domain_min[0], domain_max[0], res[0]).reshape(-1, 2)
            coords[:, 1] = domain_min[1]
            self.vertex_ids = self.participant.set_mesh_vertices(self.mesh_name, coords)

        elif self.dim == 2:
            self.mesh_name = "DataGenerator-Mesh-2D"
            self.data_name = "Data_2D_Volume"
            res = self.config["2d_resolution"]
            domain_min = self.config["domain_min"]
            domain_max = self.config["domain_max"]

            x = np.linspace(domain_min[0], domain_max[0], res[0])
            y = np.linspace(domain_min[1], domain_max[1], res[1])
            xx, yy = np.meshgrid(x, y)
            coords = np.vstack([xx.ravel(), yy.ravel()]).T
            self.vertex_ids = self.participant.set_mesh_vertices(self.mesh_name, coords)
        else:
            raise NotImplementedError("3D case is not implemented.")

    def run(self):
        dt = self.participant.get_max_time_step_size()
        while self.participant.is_coupling_ongoing():
            self.read_data()
            self.participant.advance(dt)

        self.participant.finalize()
        self.print_data()

    def read_data(self):
        read_values = self.participant.read_data(self.mesh_name, self.data_name, self.vertex_ids, 0.0)
        self.data[self.data_name] = read_values

    def print_data(self):
        print("--- DataGenerator Results ---")
        for key, value in self.data.items():
            print(f"Received data '{key}' with shape {value.shape}")

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["1", "2"]:
        print("Usage: python -m precice_datagen.datagen.datagen [1|2]")
        sys.exit(1)
    datagen = DataGenerator(int(sys.argv[1]))
    datagen.run()
