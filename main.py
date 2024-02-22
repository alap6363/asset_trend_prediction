# app.py
import streamlit as st
import yfinance as yf
from datetime import date
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objs as go

START = "2021-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Asset Trend Predictor')

stocks = ('GOOG', 'MSFT', 'GME','INFY','ETH-USD','SBCF')
selected_stock = st.selectbox('Select dataset for prediction', stocks)



@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text(' DONE!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# LSTM Model
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
df_train_scaled = scaler.fit_transform(df_train['y'].values.reshape(-1, 1))

# Create a training dataset
train_data = df_train_scaled[0:int(len(df_train_scaled)*0.95), :]
x_train, y_train = [], []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=5, batch_size=1, verbose=2)

# Test dataset
test_data = df_train_scaled[int(len(df_train_scaled)*0.95)-60:, :]

x_test = []
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Create a DataFrame with the predicted values
train = df_train[:len(df_train)-len(predictions)]
valid = df_train[len(df_train)-len(predictions):].reset_index(drop=True)
valid['Predictions'] = predictions

# Show and plot forecast
st.subheader('Forecast data with LSTM')
st.write(valid.tail())

st.write(f'Forecast plot')
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=train['ds'], y=train['y'], name='Historical Data'))
fig1.add_trace(go.Scatter(x=valid['ds'], y=valid['y'], name='Actual Data'))
fig1.add_trace(go.Scatter(x=valid['ds'], y=valid['Predictions'], name='LSTM Predictions'))
fig1.layout.update(title_text='LSTM Stock Price Prediction', xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)
