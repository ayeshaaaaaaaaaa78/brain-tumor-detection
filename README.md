# 🧠 Brain Tumor Detection from MRI Scans

A Deep Learning Computer Vision project that detects brain tumors from MRI scans using VGG16 Transfer Learning.

## 📌 Problem Statement
Brain tumors require early and accurate diagnosis. Manual MRI analysis is time-consuming and error-prone. This project automates tumor detection using Deep Learning to assist radiologists.

## 📁 Dataset
- **Source:** [Kaggle - Brain MRI Images](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Classes:** Tumor / No Tumor
- **Total Images:** 253

## ⚙️ Preprocessing
- Resized all images to 128x128
- Normalized pixel values (0-1)
- Stratified train/val/test split (70/15/15)
- Class weights to handle imbalance
- Data augmentation (rotation, flip, zoom)

## 🧠 Model
- **Base:** VGG16 (pretrained on ImageNet, frozen)
- **Head:** Flatten → Dense(256) → BatchNorm → Dropout → Dense(64) → Sigmoid
- **Optimizer:** Adam (lr=0.0001)
- **Loss:** Binary Crossentropy

## 📊 Results
| Metric | Score |
|--------|-------|
| Accuracy | 84% |
| Tumor F1 | 0.88 |
| No Tumor F1 | 0.77 |
| AUC-ROC | 0.83 |

## 🛠️ Technologies
Python, TensorFlow, Keras, OpenCV, Scikit-learn, Matplotlib

## 🚀 How to Run
Open `cv_mid.ipynb` in Google Colab and run all cells.
