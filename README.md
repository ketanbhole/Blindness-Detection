# Blindness-Detection
Overview
This repository contains a Python script for training a Convolutional Neural Network (CNN) model to detect the severity of diabetic retinopathy using retinal images. 
It also includes functionality to use the trained model to predict the severity of diabetic retinopathy for a given retinal image.

Dependencies
Python 3.9
TensorFlow
NumPy
pandas
PIL (Pillow)
scikit-learn

Usage
1. Dataset Preparation
The retinal images and their corresponding labels should be organized into specific directories or folders.
The train and test images along with their labels should be appropriately formatted and named.

You can download dataset from here:
https://www.kaggle.com/datasets/benjaminwarner/resized-2015-2019-blindness-detection-images

3. Training the Model
Modify the paths for the dataset directories in the script (train_19_labels, test_19_labels, load_images_from_folder) to match your dataset location.

Execute the script blindness_detection.py using Python.

python blindness_detection.py

The script will preprocess the data, train the CNN model, and save the trained model as blindness_detection_model.h5.

5. Prediction using Trained Model
After training, you can use the trained model for predictions on new retinal images.
Modify the image_path variable in the script blindness_detection.py to point to the location of the image you want to predict.

Run the script again:

python blindness_detection.py

The script will load the trained model, preprocess the image, and predict the severity of diabetic retinopathy for the given image.

Example Usage

from blindness_detection import predict_blindness

# Path to the user's image
image_path = r'Enter ur image path'

# Predict severity of diabetic retinopathy
predicted_severity = predict_blindness(image_path)
print(f"The predicted severity of diabetic retinopathy for the image is: {predicted_severity}")

output will show the detection in 
0 - No DR

1 - Mild

2 - Moderate

3 - Severe

4 - Proliferative DR

Note
This script assumes a specific structure and format of the dataset. Modify paths and data handling accordingly for your dataset.

Future Improvements
Enhance model architecture with different CNN architectures and hyperparameter tuning.
Implement data augmentation techniques for better model generalization.
Visualize model interpretability and optimize the model for deployment on edge devices.
Perform thorough error analysis and optimize the model's performance.
