# Kaggle Digit Recognizer: An End-to-End Classification Project

![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)

## Project Description

This repository contains the code for a complete machine learning project focused on the classic Kaggle **Digit Recognizer** competition. The goal is to build a model that can accurately identify handwritten digits (0-9) from 28x28 pixel grayscale images.

This project demonstrates a full data science workflow, including data loading, preprocessing with Principal Component Analysis (PCA), model training, and in-depth performance evaluation.

---
## Project Overview

This project covers several key machine learning concepts:
* **Data Loading & Preparation:** Loading the official Kaggle dataset and separating features from labels.
* **Preprocessing:** Scaling pixel values and applying **Principal Component Analysis (PCA)** to reduce dimensionality from 784 features to a more efficient set while retaining 95% of the data's variance.
* **Model Training:** Using a powerful `RandomForestClassifier` to learn the patterns in the digit images.
* **Robust Evaluation:** Using **K-fold cross-validation** to get a reliable estimate of the model's accuracy.
* **Performance Analysis:** Visualizing a **confusion matrix** to diagnose specific errors and plotting misclassified images for qualitative error analysis.

---
## Getting Started

No local installation is required. You can run the entire project in your browser using Google Colab.

* **Run the Notebook:**
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ItxJack/Digit-Recognizer/blob/main/YOUR_NOTEBOOK_NAME.ipynb)
---
## Dataset

The project uses the official **Digit Recognizer** dataset from Kaggle, which is a modified version of the famous MNIST dataset.
* **Training Set:** 42,000 labeled images.
* **Test Set:** 28,000 unlabeled images for submission.
* **Image Size:** 28x28 pixels (784 features).

---
## Model & Final Results

* **Preprocessing:** Principal Component Analysis (PCA) retaining 95% of variance.
* **Final Model:** `RandomForestClassifier`.
* **Performance:** The model achieves a cross-validated mean accuracy of approximately **92%** on the training data.

---
## Technologies Used

* **Python**
* **Pandas** for data manipulation.
* **NumPy** for numerical operations.
* **Scikit-Learn** for preprocessing (PCA), modeling (`RandomForestClassifier`), and evaluation.
* **Matplotlib** for data visualization.
* **Google Colab / Jupyter Notebook** for the interactive environment.

---
## License

This project is licensed under the MIT License.
