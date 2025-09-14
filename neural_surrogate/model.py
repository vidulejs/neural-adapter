import torch
import torch.nn as nn
import torch.nn.functional as F

def pad_with_ghost_cells(input_seq, bc_left, bc_right):
    return torch.cat([bc_left, input_seq, bc_right], dim=1)

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
        if output_size == input_size - 2:
            print("Assuming ghost cells")
            #assuming ghost cells are included in input_size
            self.ghost_cells = True
            self.output_size = self.input_size
            output_size = self.input_size
        
        
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

        if self.ghost_cells:
            out = out[:, 1:-1] # remove ghost cells

        return out

class ResidualBlock1D(nn.Module):
    """A residual block that uses unpadded convolutions and crops the identity."""
    def __init__(self, channels, kernel_size=3):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        crop_amount = (identity.shape[2] - out.shape[2]) // 2
        if crop_amount > 0:
            identity = identity[:, :, crop_amount:-crop_amount]
        
        out += identity
        return F.relu(out)

class CNN_RES(nn.Module):
    """The main model. It now handles internal padding."""
    def __init__(self, hidden_channels, num_blocks=2, kernel_size=3):
        super(CNN_RES, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        
        self.conv_in = nn.Conv1d(1, hidden_channels, kernel_size=1, bias=False)
        self.bn_in = nn.BatchNorm1d(hidden_channels)
        
        layers = [ResidualBlock1D(hidden_channels, kernel_size) for _ in range(num_blocks)]
        self.res_blocks = nn.Sequential(*layers)
        
        self.conv_out = nn.Conv1d(hidden_channels, 1, kernel_size=1)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) # Add channel dim: (B, 1, L)

        total_pad_per_side = self.num_blocks * (self.kernel_size - 1)
        
        internal_pad_per_side = total_pad_per_side - 1
        
        if internal_pad_per_side > 0:
            x_padded = F.pad(x, (internal_pad_per_side, internal_pad_per_side), mode='replicate')
        else:
            x_padded = x

        out = F.relu(self.bn_in(self.conv_in(x_padded)))
        out = self.res_blocks(out)
        out = self.conv_out(out)
        
        return out.squeeze(1) # Remove channel dim: (B, L_final)
    
class DoubleConv(nn.Module):
    """A helper block : (Conv2D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """
    A standard U-Net architecture for image-to-image tasks.
    It takes an input tensor and produces an output tensor of the same spatial dimensions.
    """
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder (Downsampling Path)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(2)

        # Decoder (Upsampling Path)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)

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
    def __init__(self, in_channels=9, field_channels=2):
        """
        Args:
            in_channels (int): Total number of input channels (fields + parameters).
            field_channels (int): Number of channels for which to predict a residual (e.g., velocity field no boundaries).
        """
        super().__init__()
        self.in_channels = in_channels
        self.field_channels = field_channels

        # It takes all channels as input but outputs only the field residuals.
        self.unet_backbone = UNet(in_channels=in_channels, out_channels=field_channels)

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