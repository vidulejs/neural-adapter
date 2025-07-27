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

        if self.participant.requires_initial_data():
            internal_data_values = np.zeros(len(self.internal_vertex_ids))
            boundary_data_values = internal_data_values[self.boundary_indices_in_internal_mesh]
            self.write_data(internal_data_values, boundary_data_values)

        self.participant.initialize()

    def setup_mesh(self):
        if self.dim == 1:
            self.mesh_internal_name = "Solver-Mesh-1D-Internal"
            self.mesh_boundaries_name = "Solver-Mesh-1D-Boundaries"
            self.data_name = "Data_1D"
            res = self.config["1d_resolution"]
            domain_min = self.config["1d_domain_min"]
            domain_max = self.config["1d_domain_max"]
            
            internal_coords = np.array([np.linspace(domain_min[0], domain_max[0], res[0]), np.full(res[0], domain_min[1])]).T
            self.internal_vertex_ids = self.participant.set_mesh_vertices(self.mesh_internal_name, internal_coords)

            boundary_coords = np.array([[domain_min[0], domain_min[1]], [domain_max[0], domain_max[1]]])
            self.boundary_vertex_ids = self.participant.set_mesh_vertices(self.mesh_boundaries_name, boundary_coords)
            
            internal_coord_to_index = {tuple(coord): i for i, coord in enumerate(internal_coords)}
            self.boundary_indices_in_internal_mesh = [internal_coord_to_index[tuple(coord)] for coord in boundary_coords]

        elif self.dim == 2:
            self.mesh_internal_name = "Solver-Mesh-2D-Internal"
            self.mesh_boundaries_name = "Solver-Mesh-2D-Boundaries"
            self.data_name = "Data_2D"
            res = self.config["2d_resolution"]
            domain_min = self.config["2d_domain_min"]
            domain_max = self.config["2d_domain_max"]

            x = np.linspace(domain_min[0], domain_max[0], res[0])
            y = np.linspace(domain_min[1], domain_max[1], res[1])
            xx, yy = np.meshgrid(x, y)
            internal_coords = np.vstack([xx.ravel(), yy.ravel()]).T
            self.internal_vertex_ids = self.participant.set_mesh_vertices(self.mesh_internal_name, internal_coords)

            x_coords = np.linspace(domain_min[0], domain_max[0], res[0])
            y_coords = np.linspace(domain_min[1], domain_max[1], res[1])
            
            top_boundary = np.array([[x_val, y_coords[-1]] for x_val in x_coords])
            bottom_boundary = np.array([[x_val, y_coords[0]] for x_val in x_coords])
            left_boundary = np.array([[x_coords[0], y_val] for y_val in y_coords])
            right_boundary = np.array([[x_coords[-1], y_val] for y_val in y_coords])
            
            boundary_coords = np.unique(np.concatenate((top_boundary, bottom_boundary, left_boundary, right_boundary)), axis=0)
            self.boundary_vertex_ids = self.participant.set_mesh_vertices(self.mesh_boundaries_name, boundary_coords)

            internal_coord_to_index = {tuple(coord): i for i, coord in enumerate(internal_coords)}
            self.boundary_indices_in_internal_mesh = [internal_coord_to_index[tuple(coord)] for coord in boundary_coords]
        else:
            raise NotImplementedError("3D case is not implemented.")

    def run(self):
        dt = self.participant.get_max_time_step_size()
        while self.participant.is_coupling_ongoing():
            internal_data_values = np.random.rand(len(self.internal_vertex_ids))
            boundary_data_values = internal_data_values[self.boundary_indices_in_internal_mesh]
            
            self.write_data(internal_data_values, boundary_data_values)
            self.participant.advance(dt)

        self.participant.finalize()
        print("--- Solver finished ---")

    def write_data(self, internal_data_values, boundary_data_values):
        self.participant.write_data(self.mesh_internal_name, self.data_name, self.internal_vertex_ids, internal_data_values)
        self.participant.write_data(self.mesh_boundaries_name, self.data_name, self.boundary_vertex_ids, boundary_data_values)

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["1", "2"]:
        print("Usage: python -m precice_datagen.solver.solver [1|2]")
        sys.exit(1)
    solver = Solver(int(sys.argv[1]))
    solver.run()