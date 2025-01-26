import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    """
    Attention-based fusion layer for combining features from multiple modalities.

    This layer computes attention weights for each modality (text, video, audio) and
    uses these weights to compute a weighted sum of the modality features.

    Args:
        feature_dim (int): Dimensionality of the input features for each modality.
    """
    def __init__(self, feature_dim):
        super().__init__()
        # Define attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),  # Project features to same dimension
            nn.Tanh(),                            # Apply non-linearity
            nn.Linear(feature_dim, 1),           # Compute attention scores
            nn.Softmax(dim=1)                    # Normalize scores to sum to 1
        )

    def forward(self, text_features, video_features, audio_features):
        """
        Forward pass for the attention-based fusion layer.

        Args:
            text_features (torch.Tensor): Features from the text modality, shape [batch_size, feature_dim].
            video_features (torch.Tensor): Features from the video modality, shape [batch_size, feature_dim].
            audio_features (torch.Tensor): Features from the audio modality, shape [batch_size, feature_dim].

        Returns:
            torch.Tensor: Fused features, shape [batch_size, feature_dim].
        """
        # Stack features from all modalities
        features = torch.stack([text_features, video_features, audio_features], dim=1)  # [batch_size, 3, feature_dim]
        
        # Compute attention weights
        attention_weights = self.attention(features)  # [batch_size, 3, 1]
        
        # Apply attention weights to features
        fused_features = torch.sum(features * attention_weights, dim=1)  # [batch_size, feature_dim]
        return fused_features