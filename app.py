import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Title without static hearts
st.title("Heart Disease Predictor")

# Pumping heart animation after title
heart_html = """
<style>
@keyframes beat {
  0%, 100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.3);
  }
}
.heart {
  color: red;
  font-size: 48px;
  animation: beat 1s infinite;
  display: inline-block;
  margin-left: 10px;
  vertical-align: middle;
}
</style>
<span class="heart">❤️</span>
"""

st.markdown(heart_html, unsafe_allow_html=True)

# Inputs
age = st.number_input("Enter your age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Select your sex", ['Male', 'Female'])
cp = st.selectbox("Chest pain type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting blood pressure", min_value=60, max_value=200, value=120)
chol = st.number_input("Cholesterol level", min_value=100, max_value=400, value=180)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl?", ['Yes', 'No'])
restecg = st.selectbox("Resting ECG results (0-2)", [0, 1, 2])
thalach = st.number_input("Max heart rate achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise induced angina?", ['Yes', 'No'])
oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the peak exercise ST segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of major vessels colored by fluoroscopy (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1=normal; 2=fixed defect; 3=reversible defect)", [1, 2, 3])

# Convert categorical inputs
sex_val = 1 if sex == 'Male' else 0
fbs_val = 1 if fbs == 'Yes' else 0
exang_val = 1 if exang == 'Yes' else 0

# Prepare input for model
input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs_val,
                        restecg, thalach, exang_val, oldpeak, slope, ca, thal]])

# Prediction button
if st.button("Predict", key="predict_button"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("🚨 You are at high risk of heart disease. Please consult a doctor.")
    else:
        st.success("✅ You are at low risk of heart disease.")
