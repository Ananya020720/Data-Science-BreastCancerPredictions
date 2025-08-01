import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

@st.cache_data
def load_data():
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    return df, cancer.target_names

df, target_names = load_data()

# Split data for training
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.title("Breast Cancer Classification Predictor")
st.write("This app predicts whether a breast tumor is cancerous or non-cancerous based on various features.")

# Display model accuracy
st.sidebar.title("Model Performance")
st.sidebar.metric("Accuracy", f"{accuracy:.2%}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

st.sidebar.title("Top 10 Important Features")
for i, row in feature_importance.head(10).iterrows():
    st.sidebar.write(f"{row['feature']}: {row['importance']:.3f}")

# Input section
st.header("Input Features")
st.write("Adjust the sliders to input tumor characteristics:")

col1, col2 = st.columns(2)

with col1:
    mean_radius = st.slider("Mean Radius", float(X['mean radius'].min()), float(X['mean radius'].max()), float(X['mean radius'].mean()))
    mean_texture = st.slider("Mean Texture", float(X['mean texture'].min()), float(X['mean texture'].max()), float(X['mean texture'].mean()))
    mean_perimeter = st.slider("Mean Perimeter", float(X['mean perimeter'].min()), float(X['mean perimeter'].max()), float(X['mean perimeter'].mean()))
    mean_area = st.slider("Mean Area", float(X['mean area'].min()), float(X['mean area'].max()), float(X['mean area'].mean()))
    mean_smoothness = st.slider("Mean Smoothness", float(X['mean smoothness'].min()), float(X['mean smoothness'].max()), float(X['mean smoothness'].mean()))

with col2:
    mean_compactness = st.slider("Mean Compactness", float(X['mean compactness'].min()), float(X['mean compactness'].max()), float(X['mean compactness'].mean()))
    mean_concavity = st.slider("Mean Concavity", float(X['mean concavity'].min()), float(X['mean concavity'].max()), float(X['mean concavity'].mean()))
    mean_concave_points = st.slider("Mean Concave Points", float(X['mean concave points'].min()), float(X['mean concave points'].max()), float(X['mean concave points'].mean()))
    mean_symmetry = st.slider("Mean Symmetry", float(X['mean symmetry'].min()), float(X['mean symmetry'].max()), float(X['mean symmetry'].mean()))
    mean_fractal_dimension = st.slider("Mean Fractal Dimension", float(X['mean fractal dimension'].min()), float(X['mean fractal dimension'].max()), float(X['mean fractal dimension'].mean()))

# Create input array (using mean values for remaining features)
input_features = np.zeros(len(X.columns))
feature_names = list(X.columns)

# Set the features we have sliders for
input_features[feature_names.index('mean radius')] = mean_radius
input_features[feature_names.index('mean texture')] = mean_texture
input_features[feature_names.index('mean perimeter')] = mean_perimeter
input_features[feature_names.index('mean area')] = mean_area
input_features[feature_names.index('mean smoothness')] = mean_smoothness
input_features[feature_names.index('mean compactness')] = mean_compactness
input_features[feature_names.index('mean concavity')] = mean_concavity
input_features[feature_names.index('mean concave points')] = mean_concave_points
input_features[feature_names.index('mean symmetry')] = mean_symmetry
input_features[feature_names.index('mean fractal dimension')] = mean_fractal_dimension

# Set remaining features to their mean values
for i, feature in enumerate(feature_names):
    if feature not in ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 
                      'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension']:
        input_features[i] = X[feature].mean()

# Make prediction
prediction = model.predict([input_features])
prediction_proba = model.predict_proba([input_features])

st.header("Prediction Results")
if prediction[0] == 0:
    st.success("🟢 **BENIGN** - The tumor is predicted to be benign (non-cancerous)")
    confidence = prediction_proba[0][0]
else:
    st.error("🔴 **MALIGNANT** - The tumor is predicted to be malignant (cancerous)")
    confidence = prediction_proba[0][1]

st.write(f"**Confidence:** {confidence:.2%}")

# Display prediction probabilities
st.subheader("Prediction Probabilities")
prob_df = pd.DataFrame({
    'Class': target_names,
    'Probability': prediction_proba[0]
})
st.bar_chart(prob_df.set_index('Class')) 