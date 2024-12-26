import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import pickle
from math import log10
from random import shuffle


### GPU COMPUTE
gpu_available = torch.cuda.is_available()
device = torch.device("cuda" if gpu_available else "cpu")
print(f"Using device: {device}")

### PATHS
origin_path = ""
datasets = ["cats", "dogs", "snakes"]
dataset_path = origin_path + "Animals/"
model_num = 0
model_save_file = origin_path + f"models/classification_cnn{model_num}.pth"
train_stats_file = origin_path + f"training_statistics/train_stats_cnn{model_num}.pkl"

### CONSTANTS
num_datasets = 3
dataset_size = 1000
img_name_format = "{}_{}.jpg"

in_channels = 3  # RGB format
img_size = 256
data_shape = (img_size, img_size)
output_dim = 3  # [cat, dog, snake]


### DATASET REFERENCE SHUFFLING
dataset_ref_map = [(i // dataset_size, i % dataset_size) for i in range(1, 3001)]
shuffle(dataset_ref_map)


### HYPER-PARAMETERS
learning_rate = 0.0001
batch_size = 16
epochs = 300


### CNN ARCHITECTURE

class CNN(nn.Module):
    def __init__(self, in_channels, input_shape, output_dim):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            # (256, 256, 3)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),  # (256, 256, 32)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # (128, 128, 32)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (128, 128, 64)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # (64, 64, 64)

            nn.Flatten(),  # (64 * 64 * 64,)

            nn.Linear((input_shape[0] // 4) * (input_shape[1] // 4) * 64, 128),  # (128,)
            nn.ReLU(),
            nn.Linear(128, 3)  #(3,)
            # Logits
        )

    def forward(self, x):
        return self.model(x)
    

### TENSOR CREATION and CONVERSION

def tensor2label(tensor: torch.tensor, tensor_length) -> int:
    max_ind = 0
    for i in range(tensor_length):
        if tensor[i] > tensor[max_ind]:
            max_ind = i
    return i

def label2tensor(label: int, tensor_length) -> torch.tensor:
    tensor = torch.zeros(tensor_length, dtype=torch.int8)
    tensor[label] = 1
    return tensor

def num_format(image_num: int) -> str:
    n = int(log10(image_num)) + 1
    image_num_str = str(image_num)
    image_num_str = "0" * (4-n) + image_num_str
    return image_num_str

def input_tensors(batch_idx: int, batch_size: int) -> torch.tensor:
    image_tensor = torch.zeros(batch_size, in_channels, *data_shape)
    label_tensor = torch.zeros(batch_size, 1, dtype=torch.int8)

    image_formatted = np.zeros((in_channels, *data_shape))
    for i in range(batch_size):
        img_idx = batch_idx*batch_size+i
        mapped_img_idx = dataset_ref_map[img_idx]
        path = dataset_path + datasets[mapped_img_idx[0]] + "/" + img_name_format.format(mapped_img_idx[0], num_format(mapped_img_idx[1]))
        
        image = np.array(Image.open(path), dtype = np.float32)
        image_formatted[0] = image[:, :, 0]
        image_formatted[1] = image[:, :, 1]
        image_formatted[2] = image[:, :, 2]

        image_tensor[i] = torch.tensor(image_formatted, dtype=torch.float32)
        label_tensor[i] = mapped_img_idx[0]
    image_tensor /= 255

    return image_tensor, label_tensor

input_tensors(0, 10)
