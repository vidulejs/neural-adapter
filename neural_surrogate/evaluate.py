import torch
import numpy as np
import matplotlib.pyplot as plt
from model import MLP
import os
import argparse

def evaluate(args):
    model = MLP(input_size=128, hidden_size=256, output_size=128)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    npz_files = [f for f in os.listdir(args.data_dir) if f.endswith('.npz')]
    if not npz_files:
        print("No .npz files found for evaluation.")
        return
    
    data_path = os.path.join(args.data_dir, npz_files[0])
    raw_data = np.load(data_path)['DataGenerator-Mesh-1D-Internal']

    rollout_steps = 50
    
    if raw_data.shape[0] <= rollout_steps:
        print(f"Data file {npz_files[0]} has only {raw_data.shape[0]} steps, not enough for a rollout of {rollout_steps} steps.")
        return
    start_index = np.random.randint(0, raw_data.shape[0] - rollout_steps - 1)
    
    x0_np = raw_data[start_index]
    ground_truth = raw_data[start_index : start_index + rollout_steps + 1]

    predictions = [x0_np]
    current_x = torch.from_numpy(x0_np).float().unsqueeze(0)
    
    with torch.no_grad():
        for _ in range(rollout_steps):
            y_pred = model(current_x)
            predictions.append(y_pred.squeeze().numpy())
            current_x = y_pred

    predictions = np.array(predictions)

    vmin = ground_truth.min()
    vmax = ground_truth.max()
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(predictions.T, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    plt.title('Surrogate Rollout')
    plt.xlabel('Time Step')
    plt.ylabel('Spatial Coordinate')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(ground_truth.T, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    plt.title('Ground Truth')
    plt.xlabel('Time Step')
    plt.ylabel('Spatial Coordinate')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('rollout_comparison.png')
    plt.close()
    
    mse = np.mean((predictions - ground_truth)**2)
    print(f"Corrected Rollout MSE over {rollout_steps} steps: {mse:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the neural surrogate model.')
    parser.add_argument('--data_dir', type=str, default='/home/dan/Desktop/neural-adapter/precice_datagen/datagen', help='Directory containing the training data.')
    parser.add_argument('--model_path', type=str, default='burgers_surrogate.pth', help='Path to the trained model.')
    args = parser.parse_args()
    evaluate(args)
