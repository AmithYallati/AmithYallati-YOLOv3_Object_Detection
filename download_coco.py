# download_coco.py

import os
import urllib.request
import zipfile

# Define URLs for COCO dataset
urls = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}

# Directory to save the COCO dataset
data_dir = "data"

# Create directories if they don't exist
os.makedirs(data_dir, exist_ok=True)

# Function to download and extract a file
def download_and_extract(url, extract_to):
    filename = url.split("/")[-1]
    filepath = os.path.join(data_dir, filename)

    # Download file if it does not exist
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        print(f"Downloaded {filename} successfully.")
    else:
        print(f"{filename} already exists, skipping download.")

    # Extract file
    print(f"Extracting {filename}...")
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {filename} successfully.")

# Download and extract training images
download_and_extract(urls["train_images"], os.path.join(data_dir, "train2017"))

# Download and extract validation images
download_and_extract(urls["val_images"], os.path.join(data_dir, "val2017"))

# Download and extract annotations
download_and_extract(urls["annotations"], os.path.join(data_dir, "annotations"))

print("COCO dataset downloaded and extracted successfully.")
