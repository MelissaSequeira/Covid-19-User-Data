# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load data
df = pd.read_csv("clean_covid_data.csv")

# Load trained model
model = joblib.load("covid_mortality_model.pkl")

# Page settings
st.set_page_config(page_title="COVID-19 Dashboard", layout="wide")
st.title("🦠 COVID-19 Patient Data Dashboard")

# Sidebar filters
st.sidebar.header("📊 Filter Dataset View")
gender_filter = st.sidebar.selectbox(
    "Select Gender", options=[-1, 1, 2],
    format_func=lambda x: "All" if x == -1 else ("Male" if x == 1 else "Female")
)

# Apply gender filter to dataset view
if gender_filter != -1:
    df = df[df['SEX'] == gender_filter]

# Show dataset preview
st.subheader("📋 Dataset Preview")
st.dataframe(df.head())

# Age distribution plot
st.subheader(" Age Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(df['AGE'], bins=30, kde=True, ax=ax1)
st.pyplot(fig1)

# Death distribution plot
st.subheader("Death Distribution")
fig2, ax2 = plt.subplots()
sns.countplot(x='DIED', data=df, ax=ax2)
ax2.set_xticklabels(['Survived', 'Died'])
st.pyplot(fig2)

# Diabetes vs Death plot
st.subheader("🧬 Diabetes vs Death")
fig3, ax3 = plt.subplots()
sns.countplot(x='DIABETES', hue='DIED', data=df, ax=ax3)
ax3.set_xticklabels(['No Diabetes', 'Diabetes'])
ax3.legend(title="Death", labels=['Survived', 'Died'])
st.pyplot(fig3)

# --------------------------------------------
# 🔮 Prediction Section
# --------------------------------------------

st.sidebar.header("🧪 Predict Mortality Risk")
st.sidebar.markdown("Enter patient data below:")

# Collect input for prediction
age = st.sidebar.slider("Age", 0, 100, 30)
sex = st.sidebar.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
pneumonia = st.sidebar.selectbox("Pneumonia", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
copd = st.sidebar.selectbox("COPD", [0, 1])
asthma = st.sidebar.selectbox("Asthma", [0, 1])
inmsupr = st.sidebar.selectbox("Immunosuppression", [0, 1])
hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
other_disease = st.sidebar.selectbox("Other Diseases", [0, 1])
cardio = st.sidebar.selectbox("Cardiovascular Disease", [0, 1])
obesity = st.sidebar.selectbox("Obesity", [0, 1])
renal = st.sidebar.selectbox("Chronic Renal Disease", [0, 1])
tobacco = st.sidebar.selectbox("Tobacco Use", [0, 1])
icu = st.sidebar.selectbox("Admitted to ICU", [0, 1])

# Input DataFrame
features = [
    'AGE', 'SEX', 'PNEUMONIA', 'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR',
    'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY',
    'RENAL_CHRONIC', 'TOBACCO', 'ICU'
]

input_data = pd.DataFrame([[
    age, sex, pneumonia, diabetes, copd, asthma, inmsupr, hypertension,
    other_disease, cardio, obesity, renal, tobacco, icu
]], columns=features)

# Predict button
if st.sidebar.button("🔍 Predict Mortality"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of death

    st.subheader("🧾 Prediction Result")
    st.success(f"Prediction: **{'Died' if prediction == 1 else 'Survived'}**")
    st.info(f"Probability of death: **{probability:.2%}**")
