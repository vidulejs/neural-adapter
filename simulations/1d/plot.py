import numpy as np
import matplotlib.pyplot as plt
import os
import json

def plot_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    number = input("Enter the file number:")
    data_path = os.path.join(script_dir, "datagen", "burgers_data_epoch_"+number+ ".npz")
    config_path = os.path.join(script_dir, "datagen", "config.json")

    internal_mesh_name = 'DataGenerator-Mesh-1D-Internal'

    config = json.load(open(config_path, 'r'))["datagen"]
    data = np.load(data_path)

    time_evolution_data = np.vstack(data[internal_mesh_name])

    domain_min = config["domain_min"]
    domain_max = config["domain_max"]
    resolution = config["1d_resolution"][0]

    x = np.linspace(domain_min[0], domain_max[0], resolution + 1)
    t = np.arange(time_evolution_data.shape[0] + 1)

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, x, time_evolution_data.T, cmap='viridis')

    plt.colorbar()
    plt.title('Burgers Eq.')
    plt.ylabel('Position (x)')
    plt.xlabel('Time Step')


    plt.show()

if __name__ == "__main__":
    plot_data()