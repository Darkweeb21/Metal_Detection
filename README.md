# Metal Surface Defect Detection

## Metal Surface Defect Classification using MobileNetV2

## Description
This project focuses on detecting and classifying six types of surface defects in metal sheets using MobileNetV2, a lightweight Convolutional Neural Network (CNN) model. The system preprocesses the dataset, trains the model, and evaluates its performance using key metrics like accuracy, F1-score, and confusion matrix.

## Features
- **Image Preprocessing:** Automatically resizes and structures raw dataset images.
- **MobileNetV2 Model:** Efficient deep learning model for image classification.
- **Model Saving:** Saves both the best and final trained models for future use.
- **Model Evaluation:** Provides detailed classification report, accuracy, F1-score, and a confusion matrix visualization.
- **Cross-Compatible Pathing:** Uses relative paths for GitHub and multi-device compatibility.

## Dataset
We used the **NEU Metal Surface Defects Data** which contains six types of surface defects.  
**Dataset Link:** [Download from Kaggle](https://www.kaggle.com/datasets/fantacher/neu-metal-surface-defects-data)

## Installation

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/metal-surface-defect-detection.git
   cd metal-surface-defect-detection
   ```
2, Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Usage 
Step 1: Preprocess the Data
   ```bash
   python PREPROCESS_DATA.py
   ```
Step 2: Train the Model
   ```bash
   python model.py
   ```
Step 3: Evaluate the Model
   ```bash
   python eval.py
   ```
