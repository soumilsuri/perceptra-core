import torch.nn as nn
import torchvision.models.video as video_models

class VideoEncoder(nn.Module):
    def __init__(self):
        """
        Initializes the VideoEncoder module.

        This version uses a pre-trained SlowFast model for video encoding and adds a projection layer
        to reduce the output dimensionality to 128. The backbone parameters are frozen to prevent
        them from being updated during training.

        Attributes:
            backbone (nn.Module): Pre-trained SlowFast model for video encoding.
            projection (nn.Sequential): Projection layer to reduce the output dimensionality.
        """
        super().__init__()

        # Load pre-trained SlowFast model
        self.backbone = video_models.slowfast_r50(pretrained=True)

        # Freeze backbone parameters to avoid training
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace the fully connected layer with a new sequential block
        num_fts = self.backbone.head.projection.in_features
        self.backbone.head.projection = nn.Sequential(
            nn.Linear(num_fts, 128),  # Project features to 128 dimensions
            nn.LayerNorm(128),        # Layer normalization for stability
            nn.ReLU(),                # ReLU activation for non-linearity
            nn.Dropout(0.2)           # Dropout for regularization
        )

    def forward(self, x):
        """
        Forward pass of the video encoder.

        Args:
            x (torch.tensor): Input tensor of shape [batch_size, frames, channels, height, width].

        Returns:
            torch.tensor: The encoded video representation of size (batch_size, 128).
        """
        # Transpose the input tensor to match the expected format for the backbone
        # [batch_size, frames, channels, height, width] -> [batch_size, channels, frames, height, width]
        x = x.transpose(1, 2)

        # Pass the input through the backbone and return the output
        return self.backbone(x)