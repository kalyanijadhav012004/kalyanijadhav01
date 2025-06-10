import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# Title
st.title("👤 Gender Prediction using Machine Learning")

# Load dataset
df = pd.read_csv(r"C:\Users\CSE\Desktop\ml gender gap\data.csv")

# Encode categorical column
df['C_api'] = LabelEncoder().fit_transform(df['C_api'])

# Features and Target
X = df.drop(columns=['gender'])
y = df['gender']

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Show metrics
st.subheader("📊 Model Performance")
st.write(f"✅ Accuracy: {accuracy:.4f}")
st.write(f"🎯 F1 Score (weighted): {f1:.4f}")

# Confusion Matrix
st.subheader("🔍 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# Classification Report
st.subheader("📋 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# User Input
st.subheader("🧑 Enter User Input for Gender Prediction")

input_data = {}
for col in X.columns:
    if X[col].dtype == 'object' or len(X[col].unique()) < 10:
        input_data[col] = st.selectbox(f"{col}", sorted(X[col].unique()))
    else:
        input_data[col] = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()))

# Prediction
if st.button("🔮 Predict Gender"):
    input_df = pd.DataFrame([input_data])

    # Handle encoding
    if 'C_api' in input_df.columns:
        input_df['C_api'] = LabelEncoder().fit(df['C_api']).transform(input_df['C_api'])

    prediction = model.predict(input_df)[0]
    st.success(f"📢 Predicted Gender: **{prediction}**")
