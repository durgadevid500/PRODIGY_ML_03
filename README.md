# PRODIGY_ML_03 – Cats vs Dogs Image Classification using SVM

## 📌 Project Overview
This project focuses on classifying images of cats and dogs using a Support Vector Machine (SVM) algorithm.

Histogram of Oriented Gradients (HOG) is used for feature extraction to identify important image patterns such as edges and shapes. The extracted features are then used to train the SVM model.

## 🎯 Objective
To build a Machine Learning model that can distinguish between cat and dog images using image processing and SVM classification.

## 🛠️ Technologies Used
- Python
- NumPy
- Matplotlib
- Scikit-learn
- Pillow (PIL)
- Scikit-image
- Joblib

## ⚙️ Project Workflow
1. Downloaded and loaded the Cats vs Dogs dataset.
2. Converted images into grayscale.
3. Resized images to 64 × 64 pixels.
4. Extracted image features using HOG.
5. Split the dataset into training and testing sets.
6. Trained the SVM classification model.
7. Evaluated model performance using accuracy, classification report, and confusion matrix.
8. Saved the trained model using Joblib.

## 📊 Dataset Details
- Total Images: 1,000
- Cat Images: 500
- Dog Images: 500
- Training Images: 800
- Testing Images: 200

## 📈 Model Performance

**Algorithm:** Linear Support Vector Machine (SVM)

**Feature Extraction:** Histogram of Oriented Gradients (HOG)

**Image Size:** 64 × 64

**Accuracy: 67.5%**

### Confusion Matrix

| | Predicted Cat | Predicted Dog |
|---|---:|---:|
| Actual Cat | 69 | 31 |
| Actual Dog | 34 | 66 |

## 📂 Project Structure

```text
PRODIGY_ML_03/
│
├── task3.py
├── cat_dog_hog_svm_model.pkl
└── README.md
```

## 🚀 How to Run

1. Install the required libraries:

```bash
pip install numpy matplotlib scikit-learn pillow scikit-image joblib
```

2. Download and extract the Cats vs Dogs dataset.

3. Update the dataset path in `task3
