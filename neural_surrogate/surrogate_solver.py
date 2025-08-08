import torch
import numpy as np
from model import MLP
import argparse

def run_surrogate(args):
    model = MLP(input_size=128, hidden_size=256, output_size=128)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    initial_condition = np.load(args.input_file)
    
    if initial_condition.shape[0] != 128:
        print(f"Error: Input file has incorrect shape {initial_condition.shape}. Expected (128,).")
        return

    rollout = [initial_condition]
    current_state = torch.from_numpy(initial_condition).float().unsqueeze(0)

    with torch.no_grad():
        for _ in range(args.steps):
            next_state = model(current_state)
            rollout.append(next_state.squeeze().numpy())
            current_state = next_state

    rollout = np.array(rollout)
    
    np.save(args.output_file, rollout)
    print(f"Saved rollout of shape {rollout.shape} to {args.output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the neural surrogate solver.')
    parser.add_argument('--model_path', type=str, default='burgers_surrogate.pth', help='Path to the trained model.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input file (numpy array of shape (128,)).')
    parser.add_argument('--output_file', type=str, default='surrogate_output.npy', help='Path to save the output rollout.')
    parser.add_argument('--steps', type=int, default=100, help='Number of time steps to simulate.')
    
    args = parser.parse_args()
    run_surrogate(args)