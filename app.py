import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .low-risk {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .high-risk {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DiabetesPredictor:
    def __init__(self):
        self.models = {
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'Logistic Regression': LogisticRegression(random_state=42),
            'Support Vector Machine': SVC(random_state=42, probability=True)
        }
        self.model_performance = {}
        self.trained_models = {}
        
    def generate_sample_data(self):
        """Generate sample diabetes dataset"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Pregnancies': np.random.randint(0, 17, n_samples),
            'Glucose': np.random.normal(120, 30, n_samples).astype(int),
            'BloodPressure': np.random.normal(70, 12, n_samples).astype(int),
            'SkinThickness': np.random.normal(29, 9, n_samples).astype(int),
            'Insulin': np.random.normal(155, 130, n_samples).astype(int),
            'BMI': np.random.normal(32, 8, n_samples),
            'DiabetesPedigreeFunction': np.random.uniform(0.08, 2.42, n_samples),
            'Age': np.random.randint(21, 81, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Ensure realistic ranges
        df['Glucose'] = df['Glucose'].clip(0, 200)
        df['BloodPressure'] = df['BloodPressure'].clip(40, 120)
        df['SkinThickness'] = df['SkinThickness'].clip(0, 99)
        df['Insulin'] = df['Insulin'].clip(0, 846)
        df['BMI'] = df['BMI'].clip(18, 67)
        
        # Generate target variable based on realistic patterns
        risk_score = (
            df['Glucose'] * 0.35 +
            df['BMI'] * 0.25 +
            df['Age'] * 0.15 +
            df['DiabetesPedigreeFunction'] * 0.15 +
            df['Pregnancies'] * 0.10
        )
        
        df['Outcome'] = (risk_score > risk_score.median()).astype(int)
        
        return df
    
    def train_models(self, df):
        """Train all models and store performance metrics"""
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Predictions and metrics
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.model_performance[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'true_labels': y_test,
                'model': model
            }
        
        return X_train, X_test, y_train, y_test
    
    def predict_diabetes_risk(self, model_name, input_data):
        """Predict diabetes risk for given input"""
        if model_name in self.trained_models:
            model = self.trained_models[model_name]
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            return prediction, probability
        return None, None

def main():
    # Initialize predictor
    predictor = DiabetesPredictor()
    
    # Header
    st.markdown('<h1 class="main-header">🩺 Diabetes Risk Predictor</h1>', unsafe_allow_html=True)
    
    # Generate and train on sample data
    with st.spinner('Loading and training models...'):
        df = predictor.generate_sample_data()
        X_train, X_test, y_train, y_test = predictor.train_models(df)
    
    # Sidebar for user input
    st.sidebar.header("🔍 Patient Information")
    
    # Input sliders with realistic ranges
    pregnancies = st.sidebar.slider("Pregnancies", 0, 16, 3)
    glucose = st.sidebar.slider("Glucose Level (mg/dL)", 0, 200, 120)
    blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)", 40, 120, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness (mm)", 0, 99, 29)
    insulin = st.sidebar.slider("Insulin Level (mu U/ml)", 0, 846, 155)
    bmi = st.sidebar.slider("BMI", 18.0, 67.0, 32.0)
    dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.08, 2.42, 0.75)
    age = st.sidebar.slider("Age", 21, 81, 33)
    
    # Model selection
    st.sidebar.header("🤖 ML Algorithm")
    selected_model = st.sidebar.selectbox(
        "Choose Prediction Model",
        list(predictor.models.keys())
    )
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })
    
    # Prediction button
    if st.sidebar.button("🔮 Predict Diabetes Risk", use_container_width=True):
        prediction, probability = predictor.predict_diabetes_risk(selected_model, input_data)
        
        # Display prediction result
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-box high-risk">
                ⚠️ HIGH RISK: Diabetes Detected<br>
                Confidence: {probability[1]*100:.1f}%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box low-risk">
                ✅ LOW RISK: No Diabetes Detected<br>
                Confidence: {probability[0]*100:.1f}%
            </div>
            """, unsafe_allow_html=True)
        
        # Show probability breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            fig_prob = go.Figure(data=[
                go.Bar(name='Probabilities', 
                      x=['No Diabetes', 'Diabetes'], 
                      y=[probability[0], probability[1]],
                      marker_color=['green', 'red'])
            ])
            fig_prob.update_layout(
                title="Prediction Probabilities",
                yaxis_title="Probability",
                yaxis_range=[0, 1]
            )
            st.plotly_chart(fig_prob, use_container_width=True)
        
        with col2:
            # Feature importance visualization (simplified)
            feature_importance = np.array([pregnancies, glucose, blood_pressure, 
                                         skin_thickness, insulin, bmi, dpf, age])
            feature_names = ['Pregnancies', 'Glucose', 'BP', 'Skin', 'Insulin', 'BMI', 'DPF', 'Age']
            
            fig_importance = px.bar(
                x=feature_importance,
                y=feature_names,
                orientation='h',
                title="Input Feature Values",
                labels={'x': 'Value', 'y': 'Feature'}
            )
            st.plotly_chart(fig_importance, use_container_width=True)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Overview", "📈 Model Performance", "🤖 Algorithms", "📋 Patient Stats"])
    
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.subheader("Dataset Info")
            st.write(f"Total samples: {len(df)}")
            st.write(f"Features: {len(df.columns) - 1}")
            st.write(f"Diabetes cases: {df['Outcome'].sum()} ({df['Outcome'].mean()*100:.1f}%)")
        
        with col2:
            st.subheader("Outcome Distribution")
            fig_dist = px.pie(
                values=df['Outcome'].value_counts().values,
                names=['No Diabetes', 'Diabetes'],
                color=['No Diabetes', 'Diabetes'],
                color_discrete_map={'No Diabetes':'green', 'Diabetes':'red'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Correlation heatmap
            st.subheader("Feature Correlations")
            corr_matrix = df.corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab2:
        st.header("Model Performance Comparison")
        
        # Performance metrics
        metrics_data = []
        for model_name, perf in predictor.model_performance.items():
            metrics_data.append({
                'Model': model_name,
                'Accuracy': perf['accuracy'],
                'Precision': classification_report(perf['true_labels'], perf['predictions'], 
                                                 output_dict=True)['1']['precision'],
                'Recall': classification_report(perf['true_labels'], perf['predictions'], 
                                              output_dict=True)['1']['recall'],
                'F1-Score': classification_report(perf['true_labels'], perf['predictions'], 
                                                output_dict=True)['1']['f1-score']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Metrics")
            st.dataframe(metrics_df.style.format({
                'Accuracy': '{:.3f}',
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1-Score': '{:.3f}'
            }), use_container_width=True)
        
        with col2:
            st.subheader("Accuracy Comparison")
            fig_acc = px.bar(
                metrics_df,
                x='Model',
                y='Accuracy',
                color='Accuracy',
                color_continuous_scale='Viridis',
                title="Model Accuracy Scores"
            )
            st.plotly_chart(fig_acc, use_container_width=True)
        
        # Confusion matrices
        st.subheader("Confusion Matrices")
        cols = st.columns(len(predictor.model_performance))
        
        for idx, (model_name, perf) in enumerate(predictor.model_performance.items()):
            cm = confusion_matrix(perf['true_labels'], perf['predictions'])
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['No Diabetes', 'Diabetes'],
                y=['No Diabetes', 'Diabetes'],
                title=f"{model_name}"
            )
            cols[idx].plotly_chart(fig_cm, use_container_width=True)
    
    with tab3:
        st.header("Machine Learning Algorithms")
        
        algorithms_info = {
            'K-Nearest Neighbors': {
                'description': 'Classifies based on similarity to training examples',
                'strengths': 'Simple, effective for small datasets',
                'weaknesses': 'Computationally expensive for large datasets'
            },
            'Naive Bayes': {
                'description': 'Probabilistic classifier based on Bayes theorem',
                'strengths': 'Fast, works well with high dimensions',
                'weaknesses': 'Assumes feature independence'
            },
            'Logistic Regression': {
                'description': 'Predicts probability using logistic function',
                'strengths': 'Interpretable, efficient',
                'weaknesses': 'Assumes linear decision boundary'
            },
            'Support Vector Machine': {
                'description': 'Finds optimal hyperplane for classification',
                'strengths': 'Effective in high dimensions',
                'weaknesses': 'Can be slow with large datasets'
            }
        }
        
        for algo, info in algorithms_info.items():
            with st.expander(f"📚 {algo}"):
                st.write(f"**Description**: {info['description']}")
                st.write(f"**Strengths**: {info['strengths']}")
                st.write(f"**Weaknesses**: {info['weaknesses']}")
    
    with tab4:
        st.header("Current Patient Statistics")
        
        # Create a summary of current inputs
        current_stats = {
            'Metric': ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                      'Insulin', 'BMI', 'Diabetes Pedigree', 'Age'],
            'Value': [pregnancies, glucose, blood_pressure, skin_thickness, 
                     insulin, bmi, dpf, age],
            'Normal Range': ['0-4', '<140 mg/dL', '<120/80 mm Hg', '10-40 mm', 
                           '<125 mu U/ml', '18.5-24.9', '<0.5', 'N/A']
        }
        
        stats_df = pd.DataFrame(current_stats)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # Health indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            glucose_status = "🟢 Normal" if glucose < 140 else "🟡 Borderline" if glucose < 200 else "🔴 High"
            st.metric("Glucose", f"{glucose} mg/dL", glucose_status)
        
        with col2:
            bp_status = "🟢 Normal" if blood_pressure < 120 else "🟡 Elevated" if blood_pressure < 130 else "🔴 High"
            st.metric("Blood Pressure", f"{blood_pressure} mm Hg", bp_status)
        
        with col3:
            bmi_status = "🟢 Normal" if 18.5 <= bmi <= 24.9 else "🟡 Overweight" if 25 <= bmi <= 29.9 else "🔴 Obese"
            st.metric("BMI", f"{bmi:.1f}", bmi_status)

    # Footer
    st.markdown("---")
    st.markdown(
        "**Disclaimer**: This is a demonstration application using synthetic data. "
        "For actual medical diagnosis, please consult healthcare professionals."
    )

if __name__ == "__main__":
    main()
