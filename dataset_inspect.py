from PIL import Image
import numpy as np
import math

### Dataset properties
origin_path = ""
datasets_path = origin_path + "Animals/"
datasets = ["cats", "dogs", "snakes"]
num_datasets = 3
dataset_size = 1000
img_name_format = "{}_{}.jpg"


def num_format(image_num: int) -> str:
    n = int(math.log10(image_num)) + 1
    image_num_str = str(image_num)
    image_num_str = "0" * (4-n) + image_num_str
    return image_num_str


### Image resolution and mode inspection
resolutions = []
modes = []

for dataset_num in range(num_datasets):
    path = datasets_path + datasets[dataset_num] + "/"
    for image_num in range(1, dataset_size + 1):
        try:
            image_file_name = img_name_format.format(dataset_num, num_format(image_num))
            image_path = path + image_file_name
            image = Image.open(image_path)

            # Resolution
            res = image.size
            if res not in resolutions:
                resolutions.append(res)

            # Mode
            mode = image.mode
            if mode not in modes:
                modes.append(mode)
            
        except FileNotFoundError:
            # Printing missing files to terminal
            print(f"File misssing: In {dataset_num} dataset, file number: {image_num}")

# Printing results to terminal
print("Count of different image resolutions: ", len(resolutions))
print("List of all image resolutions present:")
print(resolutions)
print()
print("Count of different image modes: ", len(modes))
print("List of all image modes present:")
print(modes)

