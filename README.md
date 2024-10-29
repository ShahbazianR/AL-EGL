# Augmented Lagrangian for Explanation-Guided Learning (ALEGL)

This repository contains Jupyter notebooks implementing the Augmented Lagrangian for Explanation-Guided Learning (ALEGL) method, designed to enhance explainability and fairness in machine learning models, especially for healthcare applications. The notebooks demonstrate the application of ALEGL to two distinct datasets for classification tasks:

- **Pneumonia X-ray Images**: Using X-ray data to classify pneumonia presence with an emphasis on interpretability through GradCAM.
- **Diabetes Prediction**: Utilizing patient health data for diabetes prediction, with improved explainability using SHAP values.

---

## Contents

1. **`Pneumonia_Augmented_Learning.ipynb`** - Notebook implementing the ALEGL method on pneumonia X-ray images. It incorporates GradCAM for visual interpretability and explanation-guided learning.
2. **`Diabetes_example.ipynb`** - Notebook applying ALEGL on the diabetes dataset. This notebook uses SHAP values to provide feature importance explanations for improved model interpretability.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ShahbazianR/ALEGL-Healthcare.git
   cd ALEGL-Healthcare
pip install numpy pandas scikit-learn tensorflow shap




# Pneumonia X-ray Classification (Pneumonia_Augmented_Learning.ipynb)
# Objective: Classify X-ray images for pneumonia presence and improve model interpretability through GradCAM heatmaps.
# Steps:
# Preprocess the X-ray dataset.
# Train a convolutional neural network (CNN) with the ALEGL framework.
# Generate GradCAM visualizations to highlight regions crucial for model predictions.
# Diabetes Prediction (Diabetes_example.ipynb)
# Objective: Predict diabetes presence using patient data while improving feature interpretability with SHAP values.
# Steps:
# Preprocess the diabetes dataset.
# Train a neural network model with ALEGL constraints to maintain fairness and interpretability.
# Use SHAP values to determine and visualize feature importance.
