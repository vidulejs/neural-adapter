import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

def pad_with_ghost_cells(input_seq, bc_left, bc_right):
    return torch.cat([bc_left, input_seq, bc_right], dim=1)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation=nn.ReLU):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

class ResidualBlockMLP(nn.Module):
    def __init__(self, size, activation=nn.ReLU):
        super(ResidualBlockMLP, self).__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.activation = activation()

    def forward(self, x):
        identity = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        return self.activation(out) + identity

class MLP_RES(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_blocks=4, activation=nn.ReLU):
        super(MLP_RES, self).__init__()
        self.activation = activation()

        # An initial projection layer to get to the hidden size
        self.fc_in = nn.Linear(input_size, hidden_size)
        
        # A sequence of residual blocks that operate on the hidden size
        layers = [ResidualBlockMLP(hidden_size, activation) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*layers)
        
        # A final layer to project to the output size
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 1. Project input to hidden dimension
        out = self.activation(self.fc_in(x))
        
        # 2. Pass through residual blocks
        out = self.res_blocks(out)
        
        # 3. Project to output dimension
        out = self.fc_out(out)
        
        return out

class LinearExtrapolationPadding1D(nn.Module):
    """Applies 'same' padding using linear extrapolation."""
    def __init__(self, kernel_size: int, dilation: int = 1):
        super().__init__()
        # Formula for 'same' padding
        self.pad_total = dilation * (kernel_size - 1)
        self.pad_beg = self.pad_total // 2
        self.pad_end = self.pad_total - self.pad_beg

    def forward(self, x):
        # Don't pad if not necessary
        if self.pad_total == 0:
            return x

        # 1. Separate the first and last actual data points.
        ghost_cell_left = x[:, :, :1]
        ghost_cell_right = x[:, :, -1:]

        # 2. Calculate the gradient at each boundary.
        grad_left = x[:, :, 1:2] - ghost_cell_left
        grad_right = ghost_cell_right - x[:, :, -2:-1]

        # grad_left = ( -11 * ghost_cell_left + 18 * x[:, :, 1:2] - 9 * x[:, :, 2:3] + 2 * x[:, :, 3:4]) / 6
        # grad_right = (11 * ghost_cell_right - 18 * x[:, :, -2:-1] + 9 * x[:, :, -3:-2] - 2 * x[:, :, -4:-3]) / 6

        # 3. Generate the extrapolated padding tensors.
        left_ramp = torch.arange(self.pad_beg, 0, -1, device=x.device, dtype=x.dtype).view(1, 1, -1)
        left_padding = ghost_cell_left - left_ramp * grad_left

        right_ramp = torch.arange(1, self.pad_end + 1, device=x.device, dtype=x.dtype).view(1, 1, -1)
        right_padding = ghost_cell_right + right_ramp * grad_right
        
        return torch.cat([left_padding, x, right_padding], dim=2)

class ResidualBlock1D(nn.Module):
    """A residual block that uses custom 'same' padding with linear extrapolation and weight normalization."""
    def __init__(self, channels, kernel_size=3, activation=nn.ReLU):
        super(ResidualBlock1D, self).__init__()
        self.activation = activation()
        # Apply weight normalization and remove bias and instance norm
        self.conv1 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding='valid', bias=True))
        self.ghost_padding1 = LinearExtrapolationPadding1D(kernel_size)
        self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding='valid', bias=True))
        self.ghost_padding2 = LinearExtrapolationPadding1D(kernel_size)

    def forward(self, x):
        identity = x
        
        out = self.ghost_padding1(x)
        out = self.conv1(out)
        out = self.activation(out)
        
        out = self.ghost_padding2(out)
        out = self.conv2(out)

        return self.activation(out) + identity
    
class CNN_RES(nn.Module):
    def __init__(self, hidden_channels, num_blocks=2, kernel_size=3, activation=nn.ReLU, ghost_cells=2):
        super(CNN_RES, self).__init__()
        self.activation = activation()
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        assert ghost_cells % 2 == 0, "ghost_cells must be even"
        self.ghost_cells = ghost_cells
        
        self.ghost_padding = LinearExtrapolationPadding1D(self.ghost_cells + self.kernel_size)

        # Apply weight normalization to the input convolution
        self.conv_in = weight_norm(nn.Conv1d(1, hidden_channels, kernel_size=1, bias=True))
        
        layers = [ResidualBlock1D(hidden_channels, kernel_size, activation=activation) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*layers)
        
        self.conv_out = nn.Conv1d(hidden_channels, 1, kernel_size=1)
    
    def forward(self, x):
        """
        Expects a pre-padded input with ghost_cells//2 number ghost cells on each side.
        Applies a custom linear extrapolation padding for inner layers.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1) # Add channel dim: (B, 1, L)

        if not self.ghost_cells == 0:
            x_padded = self.ghost_padding(x)
            
        else:
            x_padded = x

        total_pad_each_side = self.ghost_padding.pad_beg + self.ghost_cells // 2

        out = self.activation(self.conv_in(x_padded)) # no extra padding here
        out = self.res_blocks(out)
        out = self.conv_out(out) # no extra padding here

        if not self.ghost_cells == 0:
            out = out[:, :, total_pad_each_side:-total_pad_each_side] # remove ghost cells
        
        return out.squeeze(1)

class DoubleConv(nn.Module):
    """A helper block : (Conv2D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, activation=nn.ReLU):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(mid_channels),
            activation(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(out_channels),
            activation()
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """
    A standard U-Net architecture for image-to-image tasks.
    It takes an input tensor and produces an output tensor of the same spatial dimensions.
    """
    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder (Downsampling Path)
        self.inc = DoubleConv(in_channels, 64, activation=activation)
        self.down1 = DoubleConv(64, 128, activation=activation)
        self.down2 = DoubleConv(128, 256, activation=activation)
        self.down3 = DoubleConv(256, 512, activation=activation)
        
        self.pool = nn.MaxPool2d(2)

        # Decoder (Upsampling Path)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256, activation=activation)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128, activation=activation)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64, activation=activation)

        # Final output convolution
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))

        # Decoder with skip connections
        x = self.up1(x4)
        x = torch.cat([x3, x], dim=1) # Concatenate along channel dimension
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv3(x)
        
        logits = self.outc(x)
        return logits

class ResidualUNet(nn.Module):
    def __init__(self, in_channels=9, field_channels=2, activation=nn.ReLU):
        """
        Args:
            in_channels (int): Total number of input channels (fields + parameters).
            field_channels (int): Number of channels for which to predict a residual (e.g., velocity field no boundaries).
        """
        super().__init__()
        self.in_channels = in_channels
        self.field_channels = field_channels

        # It takes all channels as input but outputs only the field residuals.
        self.unet_backbone = UNet(in_channels=in_channels, out_channels=field_channels, activation=activation)

    def forward(self, x):
        """
        x: The input tensor representing State_t, with shape (N, C, H, W)
        """
        pred_residual = self.unet_backbone(x)

        # Separate the original input
        input_fields = x[:, :self.field_channels, :, :]   # first n channels
        input_params = x[:, self.field_channels:, :, :]    # remaining channels

        # Residual identity: next_field = current_field + residual
        pred_next = input_fields + pred_residual

        # Re-attach the unchanged parameter channels to form the full next state
        pred_next_full = torch.cat([pred_next, input_params], dim=1)
        
        return pred_next_full