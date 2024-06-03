# Object-Recognition-using-pre-trained-Model-MobileNetV2-
An object recognition using various pattern recognition techniques which includes some technique like Gaussian Naïve Bayes Classifier, Principal Component Analysis (PCA), Bayesian Belief Networks, and DBSCAN clustering.

## Overview
This repository contains a comprehensive case study on object recognition using various pattern recognition techniques. The study explores the implementation of different machine learning models and their applications in object recognition, image classification, and clustering.



## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Dataset Description](#dataset-description)
- [Challenges Faced](#challenges-faced)
- [Code Explanation](#code-explanation)
  - [Object Recognition](#object-recognition)
  - [Intermediate Features](#intermediate-features)
  - [Bayesian Belief Network](#bayesian-belief-network)
  - [DBSCAN Clustering](#dbscan-clustering)
  - [Dimensionality Reduction](#dimensionality-reduction)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Project Structure
- **notebooks/**: Jupyter notebooks containing code implementations
  - `casestudy_pattern_recognition.ipynb`: Main notebook for the case study
- **datasets/**: Directory for storing dataset files
- **images/**: Directory for storing result images and visualizations
- **README.md**: Project documentation

## Technologies Used
- Python
- TensorFlow
- Keras
- Numpy
- Matplotlib
- pgmpy

## Dataset Description
The dataset consists of images of various objects across different classes. The images are resized to a resolution of 224x224 pixels for processing with the MobileNetV2 model.

## Challenges Faced
- **Model Generalization**: The pre-trained MobileNetV2 model was trained on the ImageNet dataset, which is diverse and extensive.
- **Accuracy and Confidence**: The confidence scores might not always reflect the actual accuracy due to factors like image quality and background clutter.
- **Handling Multiple Objects**: Detecting and interpreting multiple objects in cluttered backgrounds.
- **Performance on Specific Objects**: Challenges in identifying objects with varying scales, orientations, or context.

## Code Explanation

### Object Recognition
1. **Import Libraries**: TensorFlow, Keras, Numpy, Matplotlib.
2. **Load Pre-trained Model**: MobileNetV2 model pre-trained on ImageNet.
3. **Preprocess Image**: Resize and preprocess images to 224x224 pixels.
4. **Prediction and Visualization**: Predict class probabilities and visualize top predictions with confidence scores.

### Intermediate Features
1. **Model Loading**: Load MobileNetV2 model.
2. **Preprocess Image**: Function to preprocess input image.
3. **Feature Extraction**: Extract intermediate features from a specified layer.
4. **Visualization**: Display intermediate features using Matplotlib.

### Bayesian Belief Network
1. **Network Creation**: Initialize a Bayesian Network with nodes and CPDs.
2. **Model Consistency**: Check model structure and CPDs.
3. **Inference**: Perform probabilistic queries using Variable Elimination.

### DBSCAN Clustering
1. **Import Libraries**: Numpy, Pandas, Matplotlib, Seaborn, DBSCAN, StandardScaler.
2. **Load Dataset**: Load data into a Pandas DataFrame.
3. **Clustering**: Apply DBSCAN on standardized features.
4. **Visualization**: Scatter plot to visualize clusters.

### Dimensionality Reduction
1. **Load Dataset**: Load Covertype dataset.
2. **Handle Missing Values**: Use SimpleImputer to replace missing values.
3. **Apply PCA**: Reduce dataset to 10 principal components.
4. **Concatenate and Display**: Combine reduced dataset with target labels and display.

## Usage
1. Clone the repository:
    ```sh
    git clone https://github.com/your_username/Pattern_Recognition_Case_Study.git
    ```
2. Navigate to the project directory:
    ```sh
    cd Pattern_Recognition_Case_Study
    ```
3. Run the Jupyter notebook:
    ```sh
    jupyter notebook notebooks/casestudy_pattern_recognition.ipynb
    ```

## Results
- **Gaussian Naïve Bayes Classifier**: Achieved an accuracy of 80.83%.
- **PCA**: Reduced dimensionality and visualized principal components.
- **Intermediate Features**: Displayed features extracted from MobileNetV2.
- **Bayesian Network**: Successfully performed inference queries.
- **DBSCAN**: Visualized clusters formed on selected features.
