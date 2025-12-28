# CIFAR-10 Image Classification using CNN

## Project Overview:
This project implements a Convolutional Neural Network (CNN) to classify real-world color images from the CIFAR-10 dataset into ten different object categories. The model learns meaningful visual patterns such as edges, textures, and shapes to accurately predict the class of unseen images.

## Problem Statement:
To build a CNN model that classifies CIFAR-10 color images into their respective object categories.

## Dataset Description:
| Feature | Description |
|--------|-------------|
| Dataset Name | CIFAR-10 |
| Total Images | 60,000 |
| Training Images | 50,000 |
| Testing Images | 10,000 |
| Image Size | 32 x 32 pixels |
| Image Type | RGB (3 Channels) |
| Number of Classes | 10 |
| Classes | Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck |


## Technologies Used:
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Model Architecture:
- Convolution + ReLU
- MaxPooling
- Convolution + ReLU
- MaxPooling
- Convolution + ReLU
- Flatten
- Dense (ReLU)
- Dense (Softmax Output Layer)

## Model Evaluation:
- Accuracy Score
- Classification Report
- Confusion Matrix
- Prediction Visualization on Test Images

## Key Observations:
- CNN effectively captures spatial features of images
- Color images require convolutional layers instead of ANN
- Model generalizes reasonably well on unseen data

## Real-World Applications:
- Face recognition systems
- Medical image diagnosis
- Autonomous vehicles
- Surveillance systems
- Product recognition in e-commerce

## Conclusion:
This project demonstrates the effectiveness of Convolutional Neural Networks for solving real-world image classification problems using the CIFAR-10 dataset. The model successfully learns hierarchical visual features and provides reliable predictions.
