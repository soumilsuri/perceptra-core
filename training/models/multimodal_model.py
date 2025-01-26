import torch
import torch.nn as nn
from text_encoder import TextEncoder
from video_encoder import VideoEncoder
from audio_encoder import AudioEncoder
from attention_fusion import AttentionFusion

class MultimodalSentimentModel(nn.Module):
    """
    Multimodal sentiment analysis model for the MELD dataset.

    This model combines features from text, video, and audio modalities using an attention-based
    fusion layer and predicts both emotion and sentiment labels.

    Attributes:
        text_encoder (nn.Module): Encoder for text modality.
        video_encoder (nn.Module): Encoder for video modality.
        audio_encoder (nn.Module): Encoder for audio modality.
        fusion_layer (nn.Module): Attention-based fusion layer for combining modalities.
        emotion_classifier (nn.Module): Classifier for predicting emotion labels.
        sentiment_classifier (nn.Module): Classifier for predicting sentiment labels.
    """
    def __init__(self):
        super().__init__()

        # Encoders for each modality
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Fusion layer
        self.fusion_layer = AttentionFusion(feature_dim=128)

        # Classification heads
        self.emotion_classifier = nn.Sequential(
            nn.Linear(128, 64),  # Project fused features to 64 dimensions
            nn.ReLU(),           # Apply non-linearity
            nn.Dropout(0.2),     # Regularization to prevent overfitting
            nn.Linear(64, 7)     # Predict 7 emotion classes
        )

        self.sentiment_classifier = nn.Sequential(
            nn.Linear(128, 64),  # Project fused features to 64 dimensions
            nn.ReLU(),           # Apply non-linearity
            nn.Dropout(0.2),     # Regularization to prevent overfitting
            nn.Linear(64, 3)     # Predict 3 sentiment classes
        )

    def forward(self, text_inputs, video_frames, audio_features):
        """
        Forward pass for the multimodal sentiment analysis model.

        Args:
            text_inputs (dict): Dictionary containing 'input_ids' and 'attention_mask' for the text modality.
            video_frames (torch.Tensor): Video frames, shape [batch_size, frames, channels, height, width].
            audio_features (torch.Tensor): Audio features, shape [batch_size, 1, 64, sequence_length].

        Returns:
            dict: Dictionary containing 'emotions' and 'sentiments' predictions.
        """
        # Extract features from each modality
        text_features = self.text_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask'],
        )  # [batch_size, 128]
        video_features = self.video_encoder(video_frames)  # [batch_size, 128]
        audio_features = self.audio_encoder(audio_features)  # [batch_size, 128]

        # Fuse multimodal features using attention
        fused_features = self.fusion_layer(text_features, video_features, audio_features)  # [batch_size, 128]

        # Predict emotions and sentiments
        emotion_output = self.emotion_classifier(fused_features)  # [batch_size, 7]
        sentiment_output = self.sentiment_classifier(fused_features)  # [batch_size, 3]

        return {
            'emotions': emotion_output,
            'sentiments': sentiment_output
        }