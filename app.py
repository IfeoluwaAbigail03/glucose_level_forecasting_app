import streamlit as st
import xgboost as xgb
import json
import numpy as np
import pandas as pd
from datetime import datetime
import random
import requests
import os
import time
import warnings


# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Glucose Predictor Pro",
    page_icon="🩸",
    layout="wide"
)

# ----------------- LOAD MODEL & FEATURES -----------------
@st.cache_resource
def load_model():
    model = xgb.Booster()
    model.load_model('best_glucose_regressor.json')
    return model

@st.cache_resource
def load_features():
    with open('feature_list.json', 'r') as f:
        return json.load(f)

model = load_model()
FEATURE_LIST = load_features()

# ----------------- SAFETY MONITORING -----------------
def is_clinically_dangerous(true_val, pred_val):
    """Identify TRULY dangerous predictions that need human review"""
    if (true_val < 4.0 and pred_val > 8.0) or (true_val > 10.0 and pred_val < 6.0):
        return True
    if abs(pred_val - true_val) > 5.0:
        return True
    return False

def safety_monitor(current_bg, prediction, patient_id):
    """Monitor predictions and flag dangerous cases"""
    risk_level = "LOW_RISK"
    alert_message = ""
    needs_human_review = False

    if abs(prediction - current_bg) > 4.0:
        risk_level = "HIGH_RISK"
        alert_message = "⚠️ Extreme swing detected! "
        needs_human_review = True

    if is_clinically_dangerous(current_bg, prediction):
        risk_level = "CRITICAL_RISK"
        alert_message = "🚨 DANGEROUS PREDICTION! "
        needs_human_review = True

    if patient_id in ['p04', 'p10']:
        risk_level = "MEDIUM_RISK" if risk_level == "LOW_RISK" else risk_level
        alert_message += f"Patient {patient_id} has historical issues. "

    return risk_level, alert_message, needs_human_review

# ----------------- SIMPLE ADVISOR -----------------
class GlucoseAdvisor:
    def __init__(self):
        self.advice_templates = {
            'critical': [
                "🚨 CRITICAL: Prediction shows {prediction:.1f} mmol/L but current is {current:.1f} mmol/L. SEEK IMMEDIATE CLINICAL REVIEW.",
                "🚨 EMERGENCY: Potential dangerous error detected. DO NOT trust this prediction. Verify with fingerstick test."
            ],
            'high': [
                "⚠️ CAUTION: Predicted {prediction:.1f} mmol/L from {current:.1f} mmol/L. Verify with fingerstick before acting.",
                "⚠️ WARNING: Large swing detected. Prediction may be unreliable. Double-check with measurement."
            ],
            'normal': [
                "✅ Predicted: {prediction:.1f} mmol/L. Good range expected. Maintain current management.",
                "📊 Prediction: {prediction:.1f} mmol/L. Levels looking stable. Continue your routine."
            ],
            'hypo': [
                "⚠️ Hypoglycemia alert: {prediction:.1f} mmol/L predicted. Consider 15g fast-acting carbs.",
                "📉 Dropping to {prediction:.1f} mmol/L. Have a snack ready to prevent low."
            ],
            'hyper': [
                "⚠️ Hyperglycemia warning: {prediction:.1f} mmol/L predicted. Monitor closely.",
                "📈 Rising to {prediction:.1f} mmol/L. Consider gentle activity or insulin as prescribed."
            ]
        }

    def get_advice(self, current_bg, prediction, risk_level):
        """Get personalized advice based on situation"""
        if risk_level == "CRITICAL_RISK":
            return random.choice(self.advice_templates['critical']).format(
                prediction=prediction, current=current_bg)

        if risk_level == "HIGH_RISK":
            return random.choice(self.advice_templates['high']).format(
                prediction=prediction, current=current_bg)

        if prediction < 4.0:
            return random.choice(self.advice_templates['hypo']).format(prediction=prediction)
        elif prediction > 10.0:
            return random.choice(self.advice_templates['hyper']).format(prediction=prediction)
        else:
            return random.choice(self.advice_templates['normal']).format(prediction=prediction)

# ----------------- MEDICAL KNOWLEDGE ADVISOR -----------------
class MedicalDiabetesAdvisor:
    def __init__(self):
        self.medical_knowledge = {
            'glucose_maintenance': {
                'questions': ['maintain', 'stable', 'control', 'manage', 'level', 'steady', 'consistent'],
                'advice': [
                    "To maintain stable glucose levels:\n\n"
                    "1. **Regular Monitoring**: Check your blood sugar 4-8 times daily\n"
                    "2. **Balanced Diet**: Consistent carb intake, high fiber, lean proteins\n"
                    "3. **Medication Adherence**: Take medications as prescribed\n"
                    "4. **Physical Activity**: 150 minutes of moderate exercise weekly\n"
                    "5. **Stress Management**: Practice relaxation techniques\n"
                    "6. **Sleep Hygiene**: 7-9 hours of quality sleep nightly\n"
                    "7. **Hydration**: Drink 8-10 glasses of water daily\n\n"
                    "Target ranges: Fasting 4.0-7.0 mmol/L, Post-meal < 10.0 mmol/L",
                    "Glucose maintenance strategies:\n\n"
                    "• **Meal Timing**: Eat at consistent times each day\n"
                    "• **Carb Counting**: Learn to estimate carbohydrate portions\n"
                    "• **Medication Timing**: Coordinate with meals and activity\n"
                    "• **Pattern Management**: Review glucose logs weekly\n"
                    "• **Healthcare Team**: Regular follow-ups with your doctor\n"
                    "• **Emergency Preparedness**: Always carry fast-acting carbs\n\n"
                    "Individual targets may vary - consult your healthcare provider"
                ]
            },
            # (you still have hypo, hyper, diet, exercise, medication dicts here – unchanged)
        }

    def get_medical_response(self, question):
        question_lower = question.lower()
        best_match = None
        highest_score = 0

        for category, data in self.medical_knowledge.items():
            score = sum(1 for keyword in data['questions'] if keyword in question_lower)
            if score > highest_score:
                highest_score = score
                best_match = category

        if best_match and highest_score > 0:
            return f"**Medical AI Assistant:** {random.choice(self.medical_knowledge[best_match]['advice'])}"

        general_advice = [
            "I specialize in diabetes management. For personalized medical advice, please consult your healthcare team.",
            "As an AI diabetes assistant, I provide general education but not personal medical advice."
        ]
        return f"**Medical AI Assistant:** {random.choice(general_advice)}"

# ----------------- MAIN PREDICTION FUNCTION -----------------
def predict_glucose(current_bg, prev_bg, trend, patient_id):
    input_features = pd.DataFrame([{
        'bg_0_00': current_bg,
        'bg_prev_1': prev_bg,
        'bg_slope_15min': trend,
        'bg_rolling_mean_3': current_bg * 0.95,
        'bg_rolling_std_3': 0.5,
        'hour': 12,
        'hypo_risk': 0,
        'roc_15min': trend,
        'roc_30min': trend * 2,
        'acceleration': 0
    }])

    for feature in FEATURE_LIST:
        if feature not in input_features.columns:
            input_features[feature] = 0

    dmatrix = xgb.DMatrix(input_features[FEATURE_LIST])
    prediction = model.predict(dmatrix)[0]

    risk_level, alert_message, needs_review = safety_monitor(current_bg, prediction, patient_id)

    advisor = GlucoseAdvisor()
    advice = advisor.get_advice(current_bg, prediction, risk_level)

    return prediction, risk_level, advice, alert_message, needs_review

# ----------------- STREAMLIT UI -----------------
st.title("🩸 Glucose Predictor Pro with Medical AI Assistant")
st.markdown("Predict future glucose levels with AI safety monitoring and medically accurate advice")

# Initialize states
if 'medical_advisor' not in st.session_state:
    st.session_state.medical_advisor = MedicalDiabetesAdvisor()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

prediction_result = None

# Sidebar Inputs
with st.sidebar:
    st.header("Patient Data")
    current_bg = st.number_input("Current Glucose (mmol/L)", min_value=2.0, max_value=20.0, value=6.0)
    prev_bg = st.number_input("Previous Glucose", min_value=2.0, max_value=20.0, value=6.2)
    trend = st.number_input("Trend (ROC 15min)", value=0.01)
    patient_id = st.text_input("Patient ID", value="p01")

    if st.button("Predict Glucose", type="primary"):
        with st.spinner("Analyzing with safety checks..."):
            prediction_result = predict_glucose(current_bg, prev_bg, trend, patient_id)

# AI Options
st.markdown("---")
st.header("AI Options")
ai_mode = st.radio(
    "Select AI Mode:",
    ["Medical Knowledge Base", "Simple Responses"],
    index=0
)

# Prediction Results
col1, col2 = st.columns(2)
with col1:
    st.header("Prediction Results")
    if prediction_result:
        prediction, risk_level, advice, alert, needs_review = prediction_result
        if risk_level == "CRITICAL_RISK":
            st.error(f"**Prediction:** {prediction:.1f} mmol/L")
        elif risk_level == "HIGH_RISK":
            st.warning(f"**Prediction:** {prediction:.1f} mmol/L")
        else:
            st.success(f"**Prediction:** {prediction:.1f} mmol/L")
        st.write(f"**Risk Level:** {risk_level}")
        if alert:
            st.warning(alert)
        if needs_review:
            st.error("**👨‍⚕️ CLINICAL REVIEW REQUIRED**")
    else:
        st.info("Enter patient data and click 'Predict Glucose'")

with col2:
    st.header("AI Health Advisor")
    if prediction_result:
        _, _, advice, _, _ = prediction_result
        st.info("💡 **AI Advice:**")
        st.write(advice)
    else:
        st.info("Prediction results will appear here")

# Chat with AI
st.markdown("---")
st.subheader("Chat with Medical AI Assistant")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about diabetes management..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        if ai_mode == "Medical Knowledge Base":
            response = st.session_state.medical_advisor.get_medical_response(prompt)
        else:
            response = "**AI Assistant:** For questions about diabetes management, please consult your healthcare provider."
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.warning("⚠️ **Disclaimer:** General diabetes education only. Not medical advice.")