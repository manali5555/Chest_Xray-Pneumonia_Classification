![IMG_1209](/IMG_1209.PNG "Optional title for IMG_1209")

# Chest_Xray-Pneumonia_Classification

## Overview
This project employs Convolutional Neural Networks (CNN) to classify chest X-ray images into two categories: presence or absence of pneumonia. It uses the Keras library with TensorFlow backend for implementing and training the model.

## Dataset
The dataset consists of chest X-ray images from different patients categorized as 'normal' or 'pneumonia'. The dataset is pre-split into training, validation, and testing sets.

## Model Architecture
The CNN model comprises several convolutional layers with ReLU activation, followed by max-pooling layers, and a final fully connected layer with a sigmoid activation function for binary classification.

## Preprocessing
Image data augmentation is used during training to improve generalization, with operations like zooming, rotation, and width/height shifting. All images are rescaled to a fixed size to match the input requirement of the CNN.

## Training
The model is trained using the Adam optimizer and binary cross-entropy loss function. A callback is implemented to save the best model based on the validation loss.

## Evaluation
Post-training, the model's performance is evaluated on a test set, and metrics such as accuracy and loss are plotted over epochs to monitor the training process.

## Results
The trained model predicts the presence of pneumonia in the test X-ray images. The prediction outputs are the probabilities given by the sigmoid function, which are then thresholded to obtain binary classifications.

## Usage
To train and evaluate the model, run the provided script. Ensure the dataset is located in the correct directory as the script expects.

## Requirements
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib

## Notes
- The script includes code to check for image path validity, ensuring all images can be read properly.
- Training can be monitored via loss and accuracy plots.
- Model checkpoints are saved in the `./saved_models/` directory.

For more information on how to set up the environment, train the model, or use the model for inference, refer to the comments within the script.

## Acknowledgements
The dataset is sourced from Kaggle.

## Author
Manali Ramchandani

