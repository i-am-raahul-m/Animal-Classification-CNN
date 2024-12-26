import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import pickle


### GPU COMPUTE
gpu_available = torch.cuda.is_available()
device = torch.device("cuda" if gpu_available else "cpu")
print(f"Using device: {device}")

### PATHS
origin_path = ""
dataset_path = origin_path + "Animals/"
model_num = 0
model_save_file = origin_path + f"models/classification_cnn{model_num}.pth"
train_stats_file = origin_path + f"training_statistics/train_stats_cnn{model_num}.pkl"

### CONSTANTS
dataset_size = 3000
