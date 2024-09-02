# download_yolov3.py

import os
import urllib.request

# Define URLs for YOLOv3 configuration and weights
yolov3_cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
yolov3_weights_url = "https://pjreddie.com/media/files/yolov3.weights"

# Directory to save the YOLOv3 files
model_dir = "model"

# Create the model directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)

# Function to download a file
def download_file(url, save_path):
    filename = url.split("/")[-1]
    filepath = os.path.join(save_path, filename)

    # Download file if it does not exist
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded {filename} successfully.")
    else:
        print(f"{filename} already exists, skipping download.")

# Download YOLOv3 configuration file
download_file(yolov3_cfg_url, model_dir)

# Download YOLOv3 weights
download_file(yolov3_weights_url, model_dir)

print("YOLOv3 configuration and weights downloaded successfully.")
