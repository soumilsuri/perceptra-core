# Import necessary libraries
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.utils.data.dataloader
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch
import subprocess
import torchaudio
from pathlib import Path

# Disable tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define a custom dataset class for MELD dataset
class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        # Load the dataset from the CSV file
        self.data = pd.read_csv(csv_path)
        
        # Directory where video files are stored
        self.video_dir = video_dir
        
        # Load the BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # Mapping of emotions to indices
        self.emotion_map = {
            'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6
        }
        
        # Mapping of sentiments to indices
        self.sentiment_map = {
            'negative': 0, 'neutral': 1, 'positive': 2
        }

    # Method to load video frames from a video file
    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            # Check if the video file can be opened
            if not cap.isOpened():
                raise ValueError(f"Video not found: {video_path}")

            # Try to read the first frame to validate the video
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found: {video_path}")

            # Reset the video capture to the beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Read up to 30 frames from the video
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize the frame and normalize pixel values
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Video error: {str(e)}")
        finally:
            # Release the video capture object
            cap.release()

        # Check if any frames were extracted
        if (len(frames) == 0):
            raise ValueError("No frames could be extracted")

        # Pad or truncate frames to ensure exactly 30 frames
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        # Convert frames to a tensor and permute dimensions
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
        

    # Method to extract audio features from a video file
    def _extract_audio_features(self, video_path):
        # Define the path for the extracted audio file
        audio_path = video_path.replace('.mp4', '.wav')

        try:
            # Use ffmpeg to extract audio from the video
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Load the extracted audio file
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample the audio if necessary
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Compute the Mel spectrogram of the audio
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )

            mel_spec = mel_spectrogram(waveform)

            # Normalize the Mel spectrogram
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            # Pad or truncate the Mel spectrogram to ensure a fixed size
            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Audio error: {str(e)}")
        finally:
            # Clean up the extracted audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)

    # Method to get the number of samples in the dataset
    def __len__(self):
        return len(self.data)

    # Method to get a sample from the dataset by index
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        row = self.data.iloc[idx]

        try:
            video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
            video_path = Path(self.video_dir) / video_filename

            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

            text_inputs = self.tokenizer(row['Utterance'], 
                                        padding='max_length', 
                                        truncation=True, 
                                        max_length=128, 
                                        return_tensors='pt')

            video_frames = self._load_video_frames(str(video_path))
            audio_features = self._extract_audio_features(str(video_path))

            emotion_label = self.emotion_map[row['Emotion'].lower()]
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]

            return {
                'text_inputs': {
                    'input_ids': text_inputs['input_ids'].squeeze(),
                    'attention_mask': text_inputs['attention_mask'].squeeze()
                },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion_label': torch.tensor(emotion_label),
                'sentiment_label': torch.tensor(sentiment_label)
            }
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            return None


# Custom collate function to filter out None samples
def collate_fn(batch):
    # Filter out None samples
    batch = list(filter(None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# Function to prepare dataloaders for training, validation, and testing
def prepare_dataloaders(train_csv, train_video_dir,
                        dev_csv, dev_video_dir,
                        test_csv, test_video_dir, batch_size=32):
    # Create dataset instances for training, validation, and testing
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)

    # Create dataloaders for each dataset
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=collate_fn)

    dev_loader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader

# Main block to test the dataloaders
if __name__ == "__main__":
    # Prepare dataloaders for training, validation, and testing
    train_loader, dev_loader, test_loader = prepare_dataloaders(
        '../dataset/train/train_sent_emo.csv', '../dataset/train/train_splits',
        '../dataset/dev/dev_sent_emo.csv', '../dataset/dev/dev_splits_complete',
        '../dataset/test/test_sent_emo.csv', '../dataset/test/output_repeated_splits_test'
    )

    # Print the shapes of the first batch of data
    for batch in train_loader:
        print("test_inputs =", batch['text_inputs'])
        print("test_video_frames =", batch['video_frames'].shape)
        print("test_audio_features =", batch['audio_features'].shape)
        print("test_emotion_label =", batch['emotion_label'])
        print("test_sentiment_label =", batch['sentiment_label'])
        break