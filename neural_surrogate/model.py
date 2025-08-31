import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

class MLP_RES(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_RES, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        self.identity_proj = torch.ones(input_size, hidden_size, requires_grad=False) if input_size != hidden_size else None
        self.identity_proj /= (hidden_size/input_size) # scaling

    def to(self, device):
        self.identity_proj = self.identity_proj.to(device) if self.identity_proj is not None else None
        return super().to(device)

    def forward(self, x):
        input = x
        
        out = F.relu(self.fc1(input))
        if self.identity_proj is not None:
            identity = input @ self.identity_proj
        out = out + identity
        
        out = F.relu(self.fc2(out))
        out = out + identity
        
        out = self.fc3(out)
        out = out + input

        return out

class ResidualBlock1D(nn.Module):
    """A standard 1D residual block."""
    def __init__(self, channels, kernel_size=3):
        super(ResidualBlock1D, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity # Residual connection
        return F.relu(out)

class CNN_RES(nn.Module):

    def __init__(self, hidden_channels, num_blocks=2, kernel_size=3):
        super(CNN_RES, self).__init__()
        # Input layer: maps 1 input channel to hidden_channels
        self.conv_in = nn.Conv1d(1, hidden_channels, kernel_size=1, bias=False)
        self.bn_in = nn.BatchNorm1d(hidden_channels)
        
        # A sequence of residual blocks operating on hidden_channels
        layers = [ResidualBlock1D(hidden_channels, kernel_size) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*layers)
        
        # Output layer: maps hidden_channels back to 1 output channel
        self.conv_out = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def add_input_channel(self, x):
        return x.unsqueeze(1)
    
    def remove_channel(self, x):
        return x.squeeze(1)

    def forward(self, x):
        out = self.add_input_channel(x)
        out = F.relu(self.bn_in(self.conv_in(out)))
        out = self.res_blocks(out)
        out = self.conv_out(out)
        return self.remove_channel(out)