# occupant_detection_classification
This repository provides code to detect people in car-cabin images and classify the type of passengers in each image (driver, front-seat passenger, back-seat passenger).


Run using: python "name".py --det "folder_containing_detections" --gt "folder_containing_ground_truth"

To perform occupant classification for the BCI_TrainingD dataset and for all images use:

"name".py -> BCI_TrainingD.py. 

"folder_containing_detections" should contain detections for each image in .txt format.

"folder_containing_ground_truth" should contain ground truth for each image in .txt format.


To perform occupant classification for the VTT1_OONZ5I dataset and for all images use:

"name".py -> VTT1_OONZ5I.py

"folder_containing_detections" should contain detections for each image in .txt format.

"folder_containing_ground_truth" should contain a file named joint.json

For day only images: "name".py -> VTT1_OONZ5I_day.py

For night only images: "name".py -> VTT1_OONZ5I_night.py


Output

For each occupant type (driver, front seat passenger, back seat passenger):

True Positives, True Negatives, False Positives, False Negatives, Accuracy

Optional Output (can be obtained by commenting out):
ROC curve, AUC, confidence thresholds that results in maximum F1 score
