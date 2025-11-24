# Diabetes Classification Project

A machine learning project that predicts diabetes risk using various classification algorithms with an interactive Streamlit web application.

## 📋 Project Overview

This project implements multiple machine learning algorithms to classify diabetes risk based on patient health metrics. The application provides an intuitive interface for users to input their health parameters and get instant diabetes risk predictions.

## 🚀 Features

- **Multiple ML Algorithms**: K-Nearest Neighbors, Naive Bayes, Logistic Regression, and Support Vector Machines
- **Interactive Web Interface**: Built with Streamlit for easy user interaction
- **Real-time Predictions**: Instant diabetes risk assessment
- **Model Performance Comparison**: Compare different algorithm performances
- **Data Visualization**: Interactive charts and metrics display

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Steps
1. Clone the repository:
```bash
git clone <your-repository-url>
cd diabetes-classification-project
Install required dependencies:

bash
pip install -r requirements.txt
Run the Streamlit app:

bash
streamlit run app.py
📊 Dataset
The project uses a diabetes dataset containing the following features:

Pregnancies

Glucose

Blood Pressure

Skin Thickness

Insulin

BMI

Diabetes Pedigree Function

Age

Outcome (Target variable)

🤖 Machine Learning Models
The project implements four classification algorithms:

K-Nearest Neighbors (KNN)

Gaussian Naive Bayes

Logistic Regression

Support Vector Machine (SVM)

🎯 Usage
Launch the application using streamlit run app.py

Input patient health parameters using the sidebar sliders

Select your preferred ML algorithm

Click "Predict Diabetes Risk" to get instant results

View model performance metrics and visualizations

📁 Project Structure
text
diabetes-classification-project/
│
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── DIABETES_CLASSIFICATION_MINIPROJECT_ML.ipynb  # Jupyter notebook with ML implementation
└── README.md             # Project documentation
📈 Model Performance
The application displays comprehensive performance metrics including:

Accuracy scores

Classification reports

Confusion matrices

Precision-Recall curves

🔧 Technical Details
Framework: Streamlit

ML Libraries: Scikit-learn, Pandas, NumPy

Visualization: Plotly, Matplotlib

Data Processing: Pandas, NumPy

👥 Contributing
Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Acknowledgments
Scikit-learn documentation

Streamlit community

Diabetes dataset providers
