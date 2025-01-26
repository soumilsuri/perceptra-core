import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer

class TextEncoder(nn.Module):
    def __init__(self):
        """
        Initializes the TextEncoder module.

        This version uses a pre-trained RoBERTa model for text encoding and adds a projection layer
        to reduce the output dimensionality to 128. The backbone parameters are frozen to prevent
        them from being updated during training.

        Attributes:
            roberta (RobertaModel): Pre-trained RoBERTa model used for encoding text.
            projection (nn.Sequential): Projection layer to reduce the output dimensionality.
        """
        super().__init__()

        # Load pre-trained RoBERTa model
        self.roberta = RobertaModel.from_pretrained('roberta-base')

        # Freeze RoBERTa parameters to avoid training
        for param in self.roberta.parameters():
            param.requires_grad = False

        # Add a projection layer to reduce the output size from 768 to 128
        self.projection = nn.Sequential(
            nn.Linear(768, 128),  # Project features to 128 dimensions
            nn.LayerNorm(128),    # Layer normalization for stability
            nn.ReLU(),            # ReLU activation for non-linearity
            nn.Dropout(0.2)       # Dropout for regularization
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the text encoder.

        Args:
            input_ids (torch.tensor): Tokenized input IDs that represent the text.
            attention_mask (torch.tensor): Mask to tell the model which tokens are actual words and which are padding.

        Returns:
            torch.tensor: The encoded text representation of size (batch_size, 128).
        """
        # Extract RoBERTa embeddings
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation (summary of the input sequence)
        pooler_output = outputs.pooler_output

        # Project the RoBERTa output to 128 dimensions
        return self.projection(pooler_output)