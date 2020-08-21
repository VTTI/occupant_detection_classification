# occupant_detection_classification
This repository provides code to detect people in car-cabin images and classify the type of passengers in each image (driver, front-seat passenger, back-seat passenger).

# Detection

The mmdetection framework (https://github.com/open-mmlab/mmdetection) was used to train and apply an object detector to detect people.
Instructions on how to use the framework are located in https://mmdetection.readthedocs.io/en/latest/.

## Installation
To install the framework the recommened method is by following these steps:

Install Docker (https://www.docker.com/).

Create a new file named "Dockerfile" and copy inside the contents of "docker/Dockerfile".

Run: docker build . -t mmdetection

Run: nvidia-docker run -it --gpus=all --rm -v "path_to_local_folder":"path_inside_docker" mmdetection bash

## Usage
An example is given in "demo/inference_demo.ipynb" (https://mmdetection.readthedocs.io/en/latest/getting_started.html) on how to run and test for a single image using a configuration file and a checkpoint file. Create a new python file copying the contents of it and run accordingly.

The configuration file used to test the Faster RCNN model is: config_faster_rcnn_x101_32x4d.py.

The checkpoint file can be found in: http://mirror.vtti.vt.edu/vtti/ctbs/passenger_detection/v1.0/faster_rcnn_x101_32x4d.pth

For further processing, output detections should be converted to .txt format with: "person confidence xmin ymin xmax ymax" on each row and a separate .txt for each image.

# Classification

## General

Install dependencies by running: pip install -r requirements_classification.txt

Run using: python "name".py --det "folder_containing_detections" --gt "folder_containing_ground_truth"

Annotations and pre-computed detections are provided in: https://dataverse.vtti.vt.edu/dataset.xhtml?persistentId=doi%3A10.15787%2FVTT1%2FWS8ORW

## BCI_TrainingD
To perform occupant classification for the BCI_TrainingD dataset and for all images use:

"name".py -> BCI_TrainingD.py. 

"folder_containing_detections" should contain detections for each image in .txt format.

"folder_containing_ground_truth" should contain ground truth for each image in .txt format (Yolo format from CVAT annotation tool https://github.com/opencv/cvat).

## VTT1_OONZ5I
To perform occupant classification for the VTT1_OONZ5I dataset and for all images use:

"name".py -> VTT1_OONZ5I.py

"folder_containing_detections" should contain detections for each image in .txt format.

"folder_containing_ground_truth" should contain a file named joint.json (COCO format from CVAT annotation tool https://github.com/opencv/cvat).

For day only images: "name".py -> VTT1_OONZ5I_day.py

For night only images: "name".py -> VTT1_OONZ5I_night.py


### Output

For each occupant type (driver, front seat passenger, back seat passenger):

True Positives, True Negatives, False Positives, False Negatives, Accuracy

Optional Output (can be obtained by commenting out):

ROC curve, AUC, confidence thresholds that results in maximum F1 score
