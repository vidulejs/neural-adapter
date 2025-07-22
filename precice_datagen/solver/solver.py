import numpy as np
import precice
import os
import sys
import json

class Solver:
    def __init__(self, dim):
        self.dim = dim
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        with open(os.path.join(script_dir, "..", "datagen", "config.json"), 'r') as f:
            self.config = json.load(f)["solver"]

        config_path = os.path.join(script_dir, "..", f"{self.dim}d", "precice-config.xml")
        self.participant = precice.Participant("Solver", config_path, 0, 1)

        self.setup_mesh()
        self.participant.initialize()
        self.data_values = np.random.rand(len(self.vertex_ids))

    def setup_mesh(self):
        if self.dim == 1:
            self.mesh_name = "Solver-Mesh-1D-Surface"
            self.data_name = "Data_1D"
            res = self.config["1d_resolution"]
            domain_min = self.config["1d_domain_min"]
            domain_max = self.config["1d_domain_max"]
            
            coords = np.linspace(domain_min[0], domain_max[0], res[0]).reshape(-1, 2)
            coords[:, 1] = domain_min[1] # Y is fixed
            self.vertex_ids = self.participant.set_mesh_vertices(self.mesh_name, coords)

        elif self.dim == 2:
            self.mesh_name = "Solver-Mesh-2D"
            self.data_name = "Data_2D_Volume"
            res = self.config["2d_resolution"]
            domain_min = self.config["2d_domain_min"]
            domain_max = self.config["2d_domain_max"]

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
            self.write_data()
            self.participant.advance(dt)

        self.participant.finalize()
        print("--- Solver finished ---")

    def write_data(self):
        self.participant.write_data(self.mesh_name, self.data_name, self.vertex_ids, self.data_values)

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["1", "2"]:
        print("Usage: python -m precice_datagen.solver.solver [1|2]")
        sys.exit(1)
    solver = Solver(int(sys.argv[1]))
    solver.run()
