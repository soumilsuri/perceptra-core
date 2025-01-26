import torch.nn as nn
import torch.nn.functional as F

class AudioEncoder(nn.Module):
    def __init__(self):
        """
        Initializes the AudioEncoder module.

        This version includes additional convolutional layers, residual connections,
        and dynamic pooling for better feature extraction.

        Attributes:
            conv_layers (nn.Sequential): Convolutional layers for extracting audio features.
            projection (nn.Sequential): Projection layer to reduce the output dimensionality.
        """
        super().__init__()

        # Define convolutional layers with residual connections
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv1d(64, 64, kernel_size=3, padding=1),  # 1D convolution with padding
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),

            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )

        # Freeze convolutional layers to avoid training
        for param in self.conv_layers.parameters():
            param.requires_grad = False

        # Define projection layer to reduce output dimensionality
        self.projection = nn.Sequential(
            nn.Linear(256, 128),  # Project features to 128 dimensions
            nn.LayerNorm(128),    # Layer normalization for stability
            nn.LeakyReLU(0.1),    # LeakyReLU for non-linearity
            nn.Dropout(0.2)       # Dropout for regularization
        )

    def forward(self, x):
        """
        Forward pass of the audio encoder.

        Args:
            x (torch.tensor): Input tensor of shape [batch_size, 1, 64, sequence_length].

        Returns:
            torch.tensor: The encoded audio representation of size (batch_size, 128).
        """
        # Remove the singleton dimension (1) from the input tensor
        x = x.squeeze(1)  # Shape: [batch_size, 64, sequence_length]

        # Extract features using the convolutional layers
        features = self.conv_layers(x)  # Shape: [batch_size, 256, 1]

        # Remove the last dimension (1) and pass through the projection layer
        return self.projection(features.squeeze(-1))  # Shape: [batch_size, 128]