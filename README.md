# PERCEPTRA-MODEL

### Model Summary Table

| Model Name               | Base Model Used                                | Input Size                        | Output Size |
|--------------------------|-----------------------------------------------|-----------------------------------|-------------|
| **TextEncoder**           | bert-base-uncased                             | (batch_size, seq_len)            | (batch_size, 128) |
| **AudioEncoder**          | Custom Audio CNN-based model                  | (batch_size, sequence_length)    | (batch_size, 128) |
| **VideoEncoder**          | facebook/r3d-18                               | (batch_size, frames, channels, height, width) | (batch_size, 128) |
| **MultimodalSentimentModel** | Combines Text, Video, and Audio Encoders | Inputs from all three encoders  | Emotion: (batch_size, 7), Sentiment: (batch_size, 3) |

---

### Detailed Explanation of Models

#### **TextEncoder**
- **Purpose**: Processes textual input using a transformer-based model.
- **Base Model**: `bert-base-uncased`, a pre-trained BERT model for text processing.
- **Techniques Used**: Parameter freezing for efficient training.
- **Processing Steps**:
  1. Load BERT as the backbone model.
  2. Freeze the parameters of BERT to prevent them from being updated during training.
  3. Extract the CLS token representation.
  4. Pass the representation through a projection layer to reduce the dimensions from 768 to 128.
- **Output**: Encoded text representation of shape (batch_size, 128).

#### **AudioEncoder**
- **Purpose**: Extracts meaningful features from speech/audio input.
- **Base Model**: Custom CNN model designed for audio processing.
- **Techniques Used**: Feature extraction using convolutional layers followed by dimensionality reduction.
- **Processing Steps**:
  1. Apply convolution layers to the audio input with two main blocks:
     - Lower level features: Conv1D(64, 64) with BatchNorm and MaxPool
     - Higher level features: Conv1D(64, 128) with BatchNorm and AdaptiveAvgPool
  2. Use batch normalization and ReLU activation for feature extraction.
  3. Apply a final projection layer with ReLU and dropout.
- **Output**: Encoded audio representation of shape (batch_size, 128).

#### **VideoEncoder**
- **Purpose**: Extracts spatiotemporal features from video frames.
- **Base Model**: `facebook/r3d-18`, a pretrained 3D ResNet model for video understanding.
- **Techniques Used**: Parameter freezing and projection layer with dropout.
- **Processing Steps**:
  1. Use pretrained R3D-18 backbone with frozen parameters.
  2. Replace final fully connected layer with a sequential block:
     - Linear layer reducing to 128 dimensions
     - ReLU activation
     - Dropout (0.2)
- **Output**: Encoded video representation of shape (batch_size, 128).

#### **MultimodalSentimentModel**
- **Purpose**: Predicts emotion and sentiment using a multimodal approach.
- **Components**:
  - Uses the **TextEncoder**, **AudioEncoder**, and **VideoEncoder** to extract representations.
  - Includes a fusion layer to combine modalities.
  - Employs separate classification heads for emotion and sentiment prediction.
- **Processing Steps**:
  1. Encode text, video, and audio inputs separately through their respective encoders.
  2. Concatenate the three 128-dimensional features into a 384-dimensional vector.
  3. Pass through fusion layer:
     - Linear(384, 256)
     - BatchNorm
     - ReLU
     - Dropout(0.3)
  4. Pass through two separate classification heads:
     - Emotion classifier: Linear(256, 64) -> ReLU -> Dropout(0.2) -> Linear(64, 7)
     - Sentiment classifier: Linear(256, 64) -> ReLU -> Dropout(0.2) -> Linear(64, 3)
- **Output**:
  - **Emotion Classification Output**: (batch_size, 7)
  - **Sentiment Classification Output**: (batch_size, 3)