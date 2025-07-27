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

        config_path = os.path.abspath(os.path.join(script_dir, "..", f"{self.dim}d", "precice-config.xml"))
        self.participant = precice.Participant("DataGenerator", config_path, 0, 1)

        self.setup_mesh()

        if self.participant.requires_initial_data():
            self.read_data()

        self.participant.initialize()
        self.data = {
            self.mesh_internal_name: [],
            self.mesh_boundaries_name: []
        }

    def setup_mesh(self):
        if self.dim == 1:
            self.mesh_internal_name = "DataGenerator-Mesh-1D-Internal"
            self.mesh_boundaries_name = "DataGenerator-Mesh-1D-Boundaries"
            self.data_name = "Data_1D"
            res = self.config["1d_resolution"]
            domain_min = self.config["domain_min"]
            domain_max = self.config["domain_max"]

            internal_coords = np.array([np.linspace(domain_min[0], domain_max[0], res[0]), np.full(res[0], domain_min[1])]).T
            self.internal_vertex_ids = self.participant.set_mesh_vertices(self.mesh_internal_name, internal_coords)

            boundary_coords = np.array([[domain_min[0], domain_min[1]], [domain_max[0], domain_max[1]]])
            self.boundary_vertex_ids = self.participant.set_mesh_vertices(self.mesh_boundaries_name, boundary_coords)

        elif self.dim == 2:
            self.mesh_internal_name = "DataGenerator-Mesh-2D-Internal"
            self.mesh_boundaries_name = "DataGenerator-Mesh-2D-Boundaries"
            self.data_name = "Data_2D"
            res = self.config["2d_resolution"]
            domain_min = self.config["domain_min"]
            domain_max = self.config["domain_max"]

            x = np.linspace(domain_min[0], domain_max[0], res[0])
            y = np.linspace(domain_min[1], domain_max[1], res[1])
            xx, yy = np.meshgrid(x, y)
            internal_coords = np.vstack([xx.ravel(), yy.ravel()]).T
            self.internal_vertex_ids = self.participant.set_mesh_vertices(self.mesh_internal_name, internal_coords)

            # Boundary vertices
            x_coords = np.linspace(domain_min[0], domain_max[0], res[0])
            y_coords = np.linspace(domain_min[1], domain_max[1], res[1])
            
            # Get the unique coordinates along the boundaries
            top_boundary = np.array([[x_val, y_coords[-1]] for x_val in x_coords])
            bottom_boundary = np.array([[x_val, y_coords[0]] for x_val in x_coords])
            left_boundary = np.array([[x_coords[0], y_val] for y_val in y_coords])
            right_boundary = np.array([[x_coords[-1], y_val] for y_val in y_coords])
            
            boundary_coords = np.unique(np.concatenate((top_boundary, bottom_boundary, left_boundary, right_boundary)), axis=0)
            
            self.boundary_vertex_ids = self.participant.set_mesh_vertices(self.mesh_boundaries_name, boundary_coords)
        else:
            raise NotImplementedError("3D case is not implemented.")

    def run(self):
        dt = self.participant.get_max_time_step_size()
        while self.participant.is_coupling_ongoing():
            self.read_data()
            self.participant.advance(dt)

        self.participant.finalize()
        self.save_data()

    def read_data(self):
        if self.dim == 1:
            internal_read_values = self.participant.read_data(self.mesh_internal_name, self.data_name, self.internal_vertex_ids, 0.0)
            self.data[self.mesh_internal_name].append(internal_read_values)
            boundary_read_values = self.participant.read_data(self.mesh_boundaries_name, self.data_name, self.boundary_vertex_ids, 0.0)
            self.data[self.mesh_boundaries_name].append(boundary_read_values)
        else:
            internal_read_values = self.participant.read_data(self.mesh_internal_name, self.data_name, self.internal_vertex_ids, 0.0)
            self.data[self.mesh_internal_name].append(internal_read_values)
            boundary_read_values = self.participant.read_data(self.mesh_boundaries_name, self.data_name, self.boundary_vertex_ids, 0.0)
            self.data[self.mesh_boundaries_name].append(boundary_read_values)

    def print_data(self):
        print("--- DataGenerator Results ---")
        for key, value in self.data.items():
            print(f"Received data '{key}' with shape {value.shape}")

    def save_data(self):
        if self.dim == 1:
            output_file = "burgers_data.npz"
            np.savez(output_file, **self.data)
            print(f"Data saved to {output_file}")
        else:
            print("Data saving for 2D is not implemented yet.")

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["1", "2"]:
        print("Usage: python -m precice_datagen.datagen.datagen [1|2]")
        sys.exit(1)
    datagen = DataGenerator(int(sys.argv[1]))
    datagen.run()