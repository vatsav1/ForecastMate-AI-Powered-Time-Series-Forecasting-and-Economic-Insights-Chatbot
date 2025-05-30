import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Forecasting Chatbot", layout="wide")

# Load datasets
unrate = pd.read_csv("UNRATE.csv", parse_dates=["observation_date"])
housing = pd.read_csv("MSPUS.csv", parse_dates=["observation_date"])
fedfunds = pd.read_csv("FEDFUNDS.csv", parse_dates=["observation_date"])

unrate.set_index("observation_date", inplace=True)
housing.set_index("observation_date", inplace=True)
fedfunds.set_index("observation_date", inplace=True)

# Title
st.title("📈 Forecasting Chatbot")
st.markdown("Forecasting economic indicators using Neural Network models.")

# User input for forecast duration
forecast_months = st.slider("Select number of months to forecast", 1, 240, 24)

# Model selection buttons
model_type = st.radio("Choose an indicator to forecast:", ['Neural Network – Unemployment Rate', 
                                                           'Neural Network – Housing Prices', 
                                                           'Neural Network – Federal Funds Rate'])

# Define a common FeedForward Neural Net class
class ForecastNN(nn.Module):
    def __init__(self, input_size=12, hidden_size=32, output_size=1):
        super(ForecastNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ------------------------
# Neural Network – Unemployment Rate
# ------------------------
if model_type == 'Neural Network – Unemployment Rate':
    model = ForecastNN()
    model.load_state_dict(torch.load("unrate_nn_model.pkl"))
    model.eval()

    # Preprocess data
    df = unrate.copy()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df["UNRATE"].values.reshape(-1, 1))

    seq_len = 12
    X = []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
    X = torch.tensor(X, dtype=torch.float32)

    future_input = X[-1].reshape(1, -1)
    future_preds = []

    with torch.no_grad():
        for _ in range(forecast_months):
            pred = model(future_input).item()
            future_preds.append([pred])
            pred_tensor = torch.tensor([[pred]], dtype=torch.float32)
            future_input = torch.cat((future_input[:, 1:], pred_tensor), dim=1)

    preds_inv = scaler.inverse_transform(future_preds)
    forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_months + 1, freq='MS')[1:]

    # Plot
    st.subheader("Forecasted U.S. Unemployment Rate (Neural Network)")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.index, df["UNRATE"], label="Historical")
    ax.plot(forecast_dates, preds_inv, label=f"{forecast_months}-Month Forecast", linestyle='--', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Unemployment Rate (%)')
    ax.set_title(f'Unemployment Rate Forecast – Next {forecast_months} Months')
    ax.legend()
    st.pyplot(fig)

# ------------------------
# Neural Network – Housing Prices
# ------------------------
elif model_type == 'Neural Network – Housing Prices':
    model = ForecastNN()
    model.load_state_dict(torch.load("mspus_nn_model.pkl"))
    model.eval()

    df = housing.copy()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df["MSPUS"].values.reshape(-1, 1))

    seq_len = 12
    X = []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
    X = torch.tensor(X, dtype=torch.float32)

    future_input = X[-1].reshape(1, -1)
    future_preds = []

    with torch.no_grad():
        for _ in range(forecast_months):
            pred = model(future_input).item()
            future_preds.append([pred])
            pred_tensor = torch.tensor([[pred]], dtype=torch.float32)
            future_input = torch.cat((future_input[:, 1:], pred_tensor), dim=1)

    preds_inv = scaler.inverse_transform(future_preds)
    forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_months + 1, freq='MS')[1:]

    # Plot
    st.subheader("Forecasted U.S. Housing Prices (Neural Network)")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.index, df["MSPUS"], label="Historical")
    ax.plot(forecast_dates, preds_inv, label=f"{forecast_months}-Month Forecast", linestyle='--', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Housing Price (USD)')
    ax.set_title(f'Housing Prices Forecast – Next {forecast_months} Months')
    ax.legend()
    st.pyplot(fig)

# ------------------------
# Neural Network – Federal Funds Rate
# ------------------------
elif model_type == 'Neural Network – Federal Funds Rate':
    model = ForecastNN()
    model.load_state_dict(torch.load("fedfunds_nn_model.pkl"))
    model.eval()

    df = fedfunds.copy()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df["FEDFUNDS"].values.reshape(-1, 1))

    seq_len = 12
    X = []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
    X = torch.tensor(X, dtype=torch.float32)

    future_input = X[-1].reshape(1, -1)
    future_preds = []

    with torch.no_grad():
        for _ in range(forecast_months):
            pred = model(future_input).item()
            future_preds.append([pred])
            pred_tensor = torch.tensor([[pred]], dtype=torch.float32)
            future_input = torch.cat((future_input[:, 1:], pred_tensor), dim=1)

    preds_inv = scaler.inverse_transform(future_preds)
    forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_months + 1, freq='MS')[1:]

    # Plot
    st.subheader("Forecasted Federal Funds Rate (Neural Network)")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.index, df["FEDFUNDS"], label="Historical")
    ax.plot(forecast_dates, preds_inv, label=f"{forecast_months}-Month Forecast", linestyle='--', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Federal Funds Rate (%)')
    ax.set_title(f'Federal Funds Rate Forecast – Next {forecast_months} Months')
    ax.legend()
    st.pyplot(fig)
