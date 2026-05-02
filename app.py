import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

# Load dataset
df = pd.read_csv("data/hr_data.csv")

# Title
st.title("💼 Employee Attrition Prediction Dashboard")

# Sidebar
st.sidebar.header("📋 Enter Employee Details")

age = st.sidebar.slider("Age", 18, 60)
monthly_income = st.sidebar.number_input("Monthly Income", 1000, 20000)
job_level = st.sidebar.selectbox("Job Level", [1, 2, 3, 4, 5])
years_at_company = st.sidebar.slider("Years at Company", 0, 40)
overtime = st.sidebar.selectbox("OverTime", [0, 1])

# -------------------------------
# 📊 ATTRITION DISTRIBUTION
# -------------------------------
st.subheader("📊 Attrition Distribution")

fig, ax = plt.subplots()
sns.countplot(x='Attrition', data=df, ax=ax)
st.pyplot(fig)

# -------------------------------
# 📈 FEATURE IMPORTANCE
# -------------------------------
st.subheader("📈 Feature Importance")

try:
    importance_df = pd.read_csv("model/feature_importance.csv")
    st.bar_chart(importance_df.set_index('Feature'))
except:
    st.warning("Feature importance file not found")

# -------------------------------
# 🎯 PREDICTION
# -------------------------------
st.subheader("🎯 Prediction")

input_data = np.array([[age, monthly_income, job_level, years_at_company, overtime]])

if st.button("Predict"):
    try:
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ High Risk: Employee likely to leave")
        else:
            st.success("✅ Low Risk: Employee likely to stay")

    except Exception as e:
        st.warning("⚠️ Input mismatch with model features. Please update input features.")
        st.text(str(e))

# -------------------------------
# 📌 SIDEBAR INFO
# -------------------------------
st.sidebar.title("📌 About")
st.sidebar.info("This app predicts employee attrition using machine learning.")