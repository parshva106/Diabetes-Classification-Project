# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# -------- CONFIG --------
MODEL_PATH = "/mnt/data/ml_model.pkl"      # change if needed
DATA_PATH  = "/mnt/data/diabetes.csv"      # change if needed

st.set_page_config(page_title="Diabetes Classifier", layout="wide")

# --------- Helpers ----------
@st.cache_data
def load_data(path: str):
    return pd.read_csv(path)

@st.cache_data
def load_model(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def predict_input(model, X):
    """
    Return predicted class and probability (if available).
    """
    pred = model.predict(X.reshape(1, -1))[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X.reshape(1, -1))[0]
    return pred, prob

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=['Non-Diabetic','Diabetic'], yticklabels=['Non-Diabetic','Diabetic'],
           ylabel='True label', xlabel='Predicted label', title='Confusion Matrix')
    # annotate
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

# --------- Main UI ----------
st.title("🩺 Diabetes Classification — Streamlit App")
st.markdown("""
This app loads a pre-trained classifier and predicts whether a person is diabetic based on clinical input features.
- Model path: `{}`  
- Data path: `{}`  
""".format(MODEL_PATH, DATA_PATH))

# Load model & data
model = None
data_load_error = None
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    data_load_error = f"Could not load model from {MODEL_PATH}: {e}"

data = None
try:
    data = load_data(DATA_PATH)
except Exception as e:
    if data_load_error:
        data_load_error += "\n"
    data_load_error = (data_load_error or "") + f"Could not load data from {DATA_PATH}: {e}"

if data_load_error:
    st.error(data_load_error)
    st.stop()

# Basic EDA / dataset preview
with st.expander("📊 Dataset preview & basic info", expanded=True):
    st.write("**First 5 rows**")
    st.dataframe(data.head())
    st.write("**Dataset shape:**", data.shape)
    if "Outcome" in data.columns:
        st.write("**Class distribution (Outcome):**")
        st.bar_chart(data["Outcome"].value_counts())

# Sidebar: user inputs
st.sidebar.header("Patient input features")

# Typical PIMA Diabetes features — if different, modify these to match your model's feature order.
def user_input_features():
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.sidebar.slider("Glucose", min_value=0, max_value=300, value=120)
    blood_pressure = st.sidebar.slider("BloodPressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.sidebar.slider("SkinThickness (mm)", min_value=0, max_value=100, value=20)
    insulin = st.sidebar.slider("Insulin (IU/mL)", min_value=0, max_value=900, value=79)
    bmi = st.sidebar.slider("BMI", min_value=0.0, max_value=80.0, value=25.0, step=0.1)
    dpf = st.sidebar.slider("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.sidebar.slider("Age", min_value=0, max_value=120, value=33)
    features = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age], dtype=float)
    return features, ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]

X_input, feature_names = user_input_features()

st.subheader("Input features (from sidebar)")
st.write(dict(zip(feature_names, X_input.tolist())))

# Prediction
if st.button("Predict"):
    try:
        pred, prob = predict_input(model, X_input)
        label_map = {0: "Non-diabetic", 1: "Diabetic"}
        st.markdown("### Prediction")
        st.write(f"**Predicted class:** {label_map.get(pred, str(pred))}")
        if prob is not None:
            # if binary classifier, show probability for positive class
            if len(prob) == 2:
                st.write(f"**Probability (Diabetic):** {prob[1]:.3f}")
                st.write(f"**Probability (Non-diabetic):** {prob[0]:.3f}")
            else:
                st.write("**Probabilities:**")
                st.write(prob)
        else:
            st.info("Model does not support `predict_proba` — only class prediction is available.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Optionally evaluate model on the provided dataset (if Outcome column exists)
if "Outcome" in data.columns:
    st.markdown("---")
    st.subheader("Model evaluation on provided CSV (quick check)")
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"].values
    # Try to ensure column order matches features used for training.
    # If your model expects different columns or preprocessing, update this part.
    try:
        y_pred = model.predict(X)
        report = classification_report(y, y_pred, output_dict=True)
        st.write("**Classification report (on CSV):**")
        st.write(pd.DataFrame(report).transpose())
        st.pyplot(plot_confusion_matrix(y, y_pred))
    except Exception as e:
        st.warning("Could not evaluate model on CSV — model may require preprocessing or different column order. Error: " + str(e))

# Download example input as CSV
st.markdown("---")
st.subheader("Export / Download")
if st.button("Download sample inputs (CSV)"):
    # create a small sample from dataset or use the input
    sample = data.head(10)
    csv = sample.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download dataset sample (CSV)", data=csv, file_name="diabetes_sample.csv", mime="text/csv")

st.markdown("""
---
**Notes & Tips**
- If your model requires preprocessing (scaling, encoding) you must apply the same preprocessing before prediction.  
  If your `ml_model.pkl` already contains a pipeline (scaler + model) this will work out-of-the-box.  
- If the feature order differs from the PIMA order used here, modify the sidebar inputs and the `feature_names` list accordingly.
- Run: `streamlit run app.py`
""")
