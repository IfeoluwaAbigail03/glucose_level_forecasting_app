# Live demo:
https://glucoselevelforecastingapp-q6ph98ipfnu3hjhxrvmtag.streamlit.app/
# Glucose Predictor Pro
A glucose prediction system with safety monitoring and general advisor.
## Features
- **Prediction**: XGBoost model for glucose forecasting
- **Safety Monitoring**: Detects and flags dangerous predictions (2.64% error cases)
- **Clinical Advisor**: Rule-based chatbot for medical guidance
- **Real-time Alerts**: Flags predictions needing human review

## Model Performance
- **Overall MAE**: 0.94 mmol/L
- **Clinical Accuracy**: 97.36% safe predictions
- **Dangerous Errors**: 2.64% (monitored and flagged)

## Installation
1. Clone the repository:
```bash
git clone https://github.com/IfeoluwaAbigail03/glucose_level_forecasting_app.git
cd glucose_level_forecasting_app
