# 🩺 Diabetes Classification — Machine Learning Project

This project predicts whether a patient is diabetic based on medical diagnostic measurements using Machine Learning.  
The application includes a Streamlit-based user interface for interactive predictions.

---

## 📘 Project Overview

The goal of this project is to build a classification model that can accurately predict diabetes using patient medical attributes such as glucose level, BMI, age, blood pressure, etc.  

The complete workflow includes:
- Data preprocessing  
- Exploratory Data Analysis (EDA)  
- Model training & evaluation  
- Saving the trained model  
- Deploying the model using Streamlit  

---

## 📂 Repository Structure

```

├── app.py                                   # Streamlit Web Application
├── diabetes.csv                             # Dataset used for training/testing
├── ml_model.pkl                             # Trained Machine Learning model
├── DIABETES_CLASSIFICATION_MINIPROJECT_ML.ipynb   # Jupyter Notebook with training workflow
└── README.md                                 # Project Documentation

````

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone <your-repo-link>
cd <repository-folder>
````

### 2️⃣ Install dependencies

Ensure Python 3.8+ is installed.

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit App

```bash
streamlit run app.py
```

Your application will open at:

```
http://localhost:8501
```

---

## 🧠 Machine Learning Details

### **Dataset Used**

PIMA Indians Diabetes Dataset

* **Target Column:** `Outcome`
* 1 → Diabetic
* 0 → Non-Diabetic

### **Modeling Steps**

* Missing value handling
* Outlier checking
* Normalization (if required)
* Splitting into train/test
* Training ML classification models
* Saving the best model (`ml_model.pkl`)

### **Evaluation Metrics**

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

---

## 🖥 Streamlit Application Features

✔ Sidebar inputs for medical features
✔ Predicts whether the patient is *Diabetic* or *Non-Diabetic*
✔ Shows probability score (if model supports it)
✔ Displays dataset preview
✔ Visualizes class distribution
✔ Shows classification report & confusion matrix

---

## 📊 Example Medical Inputs Used

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age

---

