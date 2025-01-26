import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
from sklearn.metrics import precision_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

from training.meld_dataset import MELDDataset
from training.models.text_encoder import TextEncoder
from training.models.video_encoder import VideoEncoder
from training.models.audio_encoder import AudioEncoder
from training.models.attention_fusion import AttentionFusion
from training.models.multimodal_model import MultimodalSentimentModel