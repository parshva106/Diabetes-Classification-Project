import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Try importing sklearn safely
try:
    from sklearn.metrics import confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except:
    SKLEARN_AVAILABLE = False

MODEL_PATH = "ml_model.pkl"
DATA_PATH = "diabetes.csv"

st.set_page_config(page_title="Diabetes Classifier", layout="wide")
st.title("🩺 Diabetes Classification — Streamlit App")

# ------------------ LOADING ------------------
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

model = load_model()
data = load_data()

# ------------------ SIDEBAR INPUTS ------------------
st.sidebar.header("Patient Input Features")

def user_inputs():
    pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
    glucose = st.sidebar.slider("Glucose", 0, 300, 120)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 200, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
    insulin = st.sidebar.slider("Insulin", 0, 900, 79)
    bmi = st.sidebar.slider("BMI", 0.0, 80.0, 25.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.sidebar.slider("Age", 1, 120, 33)

    return np.array([pregnancies, glucose, blood_pressure,
                     skin_thickness, insulin, bmi, dpf, age])

user_input = user_inputs()

# ------------------ PREDICTION ------------------
if st.button("Predict"):

    if model is None:
        st.error("Model not loaded.")
    else:
        pred = model.predict(user_input.reshape(1, -1))[0]
        result = "Diabetic" if pred == 1 else "Non-Diabetic"

        st.subheader("Prediction Result:")
        st.write(f"### 🟢 {result}")

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(user_input.reshape(1, -1))[0]
            st.write(f"Probability (Diabetic): **{prob[1]:.3f}**")
            st.write(f"Probability (Non-Diabetic): **{prob[0]:.3f}**")

# ------------------ DATASET PREVIEW ------------------
if data is not None:
    with st.expander("📊 Dataset Overview", expanded=True):
        st.write(data.head())
        st.write("Shape:", data.shape)
        st.bar_chart(data["Outcome"].value_counts())

# ------------------ EVALUATION ------------------
if SKLEARN_AVAILABLE and model is not None and data is not None:
    X = data.drop("Outcome", axis=1)
    y_true = data["Outcome"]
    y_pred = model.predict(X)

    with st.expander("📈 Model Evaluation"):
        st.write(pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).T)

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

elif not SKLEARN_AVAILABLE:
    st.warning("⚠ scikit-learn not installed. Evaluation disabled.")
