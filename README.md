# ForecastMate: AI-Powered Economic Forecasting Chatbot

## 📈 Project Overview

ForecastMate is a time series forecasting chatbot built using Python, Streamlit, and PyTorch. It provides accurate, user-interactive predictions for crucial economic indicators, including:

- **U.S. Unemployment Rate (UNRATE)**
- **Median Housing Prices (MSPUS)**
- **Federal Funds Rate (FEDFUNDS)**

The system utilizes multiple forecasting models:
- ARIMA
- SARIMA
- Feedforward Neural Networks (FFNN)
- Long Short-Term Memory Networks (LSTM)

Users can:
- Choose economic indicators
- Define custom forecast durations (e.g., 24 months)
- View real-time predictions and trend visualizations
- Compare model accuracy across economic crises (e.g., 2008 Financial Crisis, COVID-19)

---

## 🧠 Key Features

✅ Streamlit-based chatbot interface  
✅ User-controlled forecasting horizon  
✅ Neural network models trained using PyTorch  
✅ Stationarity checks (ADF test) and autocorrelation analysis (ACF, PACF)  
✅ Model comparison using error metrics: MAE, RMSE, MAPE  
✅ Residual diagnostics and density plot analysis  
✅ Visualization of actual vs predicted trends  
✅ Event-specific forecasting (2008 Recession, COVID-19)

---

## 🗃️ Project Structure

forecastmate/
│
├── data/
│ ├── UNRATE.csv
│ ├── MSPUS.csv
│ └── FEDFUNDS.csv
│
├── models/
│ ├── unrate_nn_model.pkl
│ ├── fedfunds_nn_model.pkl
│ └── mspus_nn_model.pkl
│
├── notebooks/
│ ├── forecasting_project-V3.ipynb
│ ├── projectV3_UnemploymentRates.ipynb
│ └── ... (notebooks with ARIMA/SARIMA tuning)
│
├── chatbot_app.py
└── README.md

Run the chatbot:
python -m streamlit run chatbot_app.py 
