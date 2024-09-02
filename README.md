
# YOLOv3 Object Detection

## Overview
This project implements a YOLOv3-based object detection system. YOLOv3 (You Only Look Once, Version 3) is a powerful, real-time object detection algorithm capable of detecting multiple objects within an image or video frame. This repository includes scripts for training the YOLOv3 model on the COCO dataset, as well as scripts for performing object detection using the trained model.

## Project Structure

Here is a detailed breakdown of the project directory structure:

E:\yolov3_object_detection\
├── data\
│   ├── annotations\
│   │   ├── annotations_trainval2017.zip
│   ├── images\
│   │   ├── train2017.zip
│   │   ├── val2017.zip
├── model\
│   ├── yolov3.cfg
│   ├── yolov3.weights
├── output\
│   ├── (Annotated images and detection results will be saved here)
├── scripts\
│   ├── detect.py
│   ├── train.py
│   ├── yolo_model.py
│   ├── yolo_model.h5
│   ├── __pycache__/
├── source\
│   ├── Include\
│   ├── Lib\
│   ├── Scripts\
│   │   ├── activate
│   │   ├── activate.bat
│   │   ├── Activate.ps1
│   │   ├── deactivate.bat
│   │   ├── pip.exe
│   │   ├── pip3.12.exe
│   │   ├── pip3.exe
│   │   ├── python.exe
│   │   ├── pythonw.exe
│   ├── pyvenv.cfg
├── download_coco.py
├── download_yolov3.py
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore

### Folder and File Details

- **`data/`**: Contains datasets required for training and validation.
  - **`annotations/`**: Contains annotation files for object detection.
    - `annotations_trainval2017.zip`: Contains the annotations (bounding box coordinates and class labels) for training and validation images.
  - **`images/`**: Contains image files for training and validation.
    - `train2017.zip`: Contains training images.
    - `val2017.zip`: Contains validation images.

- **`model/`**: Contains model configuration and weights files.
  - `yolov3.cfg`: Configuration file for the YOLOv3 model, including network architecture and parameters.
  - `yolov3.weights`: Pre-trained weights file for the YOLOv3 model.

- **`output/`**: Directory where output files from object detection are saved.
  - Annotated images and detection results will be stored here after running the detection script.

- **`scripts/`**: Contains Python scripts for training and detecting objects.
  - `detect.py`: Script for running object detection on new images or video streams.
  - `train.py`: Script for training the YOLOv3 model on the COCO dataset.
  - `yolo_model.py`: Contains model definition and utility functions.
  - `yolo_model.h5`: Pre-trained model file (if applicable).
  - `__pycache__/`: Directory containing cached bytecode files (auto-generated).

- **`source/`**: Contains Python virtual environment files and dependencies.
  - **`Include/`**: Includes headers for the Python virtual environment.
  - **`Lib/`**: Contains library files for the Python virtual environment.
  - **`Scripts/`**: Contains scripts to manage the virtual environment.
    - `activate`: Script to activate the virtual environment (for Unix-like systems).
    - `activate.bat`: Script to activate the virtual environment (for Windows).
    - `Activate.ps1`: PowerShell script to activate the virtual environment (for Windows).
    - `deactivate.bat`: Script to deactivate the virtual environment (for Windows).
    - `pip.exe`, `pip3.12.exe`, `pip3.exe`: Executables for package management.
    - `python.exe`, `pythonw.exe`: Python executables.
  - `pyvenv.cfg`: Configuration file for the Python virtual environment.

- **`download_coco.py`**: Script to automate the download of the COCO dataset.

- **`download_yolov3.py`**: Script to automate the download of YOLOv3 weights and configuration files.

- **`README.md`**: Markdown file containing the project overview, instructions, and documentation.

- **`requirements.txt`**: File listing the Python packages required for the project.

- **`LICENSE`**: License file detailing the project's licensing terms.

- **`.gitignore`**: File specifying which files and directories to ignore in version control.

## COCO Dataset
The COCO (Common Objects in Context) dataset is used for training and evaluating the YOLOv3 model. COCO is a large-scale object detection, segmentation, and captioning dataset. It contains over 200,000 labeled images with more than 80 object categories. Below are the links to download the necessary parts of the COCO dataset:

- **COCO 2017 Train/Val Annotations [241MB]**: Contains the annotations (bounding box coordinates and class labels) for both training and validation images.
  - [Download COCO 2017 Train/Val Annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

- **COCO 2017 Training Images [118K/18GB]**: Contains around 118,000 images for training purposes.
  - [Download COCO 2017 Training Images](http://images.cocodataset.org/zips/train2017.zip)

- **COCO 2017 Validation Images [5K/1GB]**: Contains around 5,000 images for validating the model's performance.
  - [Download COCO 2017 Validation Images](http://images.cocodataset.org/zips/val2017.zip)

After downloading, unzip these files into the `data/` directory. The directory structure should look like this:

YOLOv3_Object_Detection/
├── data/
│   ├── annotations/
│   │   ├── annotations_trainval2017.zip
│   ├── images/
│   │   ├── train2017.zip
│   │   ├── val2017.zip
├── model/
│   ├── yolov3.cfg
│   ├── yolov3.weights
├── scripts/
│   ├── detect.py
│   ├── train.py
│   ├── yolo_model.py
│   ├── yolo_model.h5
│   ├── __pycache__/
├── download_coco.py
├── download_yolov3.py
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore

## YOLOv3 Model Files

- **`yolov3.cfg`**: This file contains the configuration of the YOLOv3 model, including the network architecture, number of classes, and anchor boxes.

- **`yolov3.weights`**: This file contains the pre-trained weights of the YOLOv3 model, which can be used for inference or fine-tuning.

To download these model files, you can run the `download_yolov3.py` script provided in this repository. This script will automatically download the YOLOv3 weights and configuration files and save them in the `model/` directory.

## Getting Started

To get started with this project:

1. **Clone the Repository**: Clone the repository to your local machine.
   ```bash
   git clone https://github.com/AmithYallati/AmithYallati-YOLOv3_Object_Detection.git
   ```

2. **Install Dependencies**: Install the required Python packages such as TensorFlow, Keras, OpenCV, and NumPy. You can install these dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download COCO Dataset**: Download and unzip the COCO dataset into the `data/` directory as specified above. Use the `download_coco.py` script to automate the download process:
   ```bash
   python download_coco.py
   ```

4. **Download YOLOv3 Weights**: Download the pre-trained YOLOv3 weights file by running:
   ```bash
   python download_yolov3.py
   ```

5. **Train the Model**: Use the `train.py` script to train the model on the COCO dataset:
   ```bash
   python scripts/train.py
   ```

6. **Run Object Detection**: Use the `detect.py` script to perform object detection on new images or video streams:
   ```bash
   python scripts/detect.py
   ```

## Usage

- **Training**: Run the `train.py` script to train the YOLOv3 model with your dataset.

- **Detection**: Run the `detect.py` script to detect objects in an input image using the trained model.

## Output

When running the `detect.py` script, the output will include:

- **Annotated Images**: Images with detected objects highlighted by
