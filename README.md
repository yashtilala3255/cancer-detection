# 🧠 Skin Cancer Detection using Deep Learning

## 📌 Project Overview

This project focuses on detecting skin cancer (melanoma) using deep learning techniques. The system analyzes skin lesion images and classifies them as **cancerous or non-cancerous**, helping in early diagnosis and improving medical decision-making.

---

## 🎯 Objectives

* Build an automated system for skin cancer detection
* Classify images using Convolutional Neural Networks (CNN)
* Improve diagnosis speed and accuracy
* Provide a simple interface for users to upload images and get predictions

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy & Pandas
* Matplotlib / Seaborn
* Scikit-learn
* Streamlit (for web app)

---

## 📊 Dataset

* Dataset sourced from Kaggle (SIIM-ISIC Melanoma Classification)
* Contains thousands of labeled skin lesion images
* Includes both **benign (non-cancerous)** and **malignant (cancerous)** samples
* Dataset has class imbalance, handled using data augmentation techniques

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Resize images to fixed dimensions
* Normalize pixel values
* Split into training and testing datasets

### 2. Data Augmentation

* Rotation, flipping, zooming
* Brightness and contrast adjustments
* Helps reduce overfitting

### 3. Model Building

* Convolutional Neural Network (CNN)
* Layers: Conv2D, MaxPooling, Dense, Dropout
* Binary classification (Cancer / No Cancer)

### 4. Model Training

* Optimizer: Adam
* Loss Function: Binary Crossentropy
* Evaluation using accuracy and validation loss

### 5. Prediction System

* User uploads image
* Model processes image
* Output: Cancerous / Non-Cancerous

---

## 🚀 How to Run the Project

### 1. Install Dependencies

```
pip install -r requirements.txt
```

### 2. Train Model

```
python train.py
```

### 3. Run Web App

```
streamlit run app.py
```

---

## 📈 Expected Output

* Input: Skin lesion image
* Output:

  * ✅ Non-Cancerous
  * ⚠️ Cancerous

---

## ⚠️ Limitations

* Requires good quality dataset
* Accuracy depends on training data
* Not a replacement for professional medical diagnosis

---

## 🔮 Future Improvements

* Use Transfer Learning (ResNet, EfficientNet)
* Improve dataset balance
* Deploy on cloud for real-time usage
* Add multi-class cancer classification

---

## 👨‍💻 Author

**Yash Tilala**

---

## 📌 Note

This project is for educational purposes and should not be used as a medical diagnostic tool.
