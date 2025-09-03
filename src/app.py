import streamlit as st
import pickle
import os
import pandas as pd

# -------------------------------
# Load trained model
# -------------------------------
model_path = os.path.join("models", "doctor.pkl")
with open(model_path, 'rb') as file:
    data = pickle.load(file)

model = data['model']
encoder = data['encoder']
symptom_map = data['symptom_map']
symptoms = list(symptom_map.keys())

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Doctor", page_icon="🩺", layout="wide")

st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #2E86C1;
        }
        .sub-title {
            text-align: center;
            font-size: 18px;
            color: gray;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🩺 AI Doctor - Smart Disease Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter your symptoms and let AI assist you with a quick diagnosis.</div>', unsafe_allow_html=True)
st.write("")

# -------------------------------
# User Input
# -------------------------------
selected_symptoms = st.multiselect(
    "🔍 Select your symptoms (searchable):",
    options=symptoms,
    help="Type to search and select multiple symptoms."
)

if st.button("🔮 Predict"):
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom to continue.")
    else:
        input_vector = [0] * len(symptom_map)
        for symptom in selected_symptoms:
            input_vector[symptom_map[symptom]] = 1

        prediction_encoded = model.predict([input_vector])[0]
        predicted_disease = encoder.inverse_transform([prediction_encoded])[0]

        st.success(f"🤖 **Predicted Disease:** {predicted_disease}")

        st.info("💡 *This is a preliminary prediction. Consult a certified doctor for medical advice.*")
