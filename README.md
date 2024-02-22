# Asset Trend Predictor

## Overview

The Asset Trend Predictor is a Streamlit web application that leverages LSTM (Long Short-Term Memory) neural networks to predict the future trends of selected assets. The application fetches historical stock or cryptocurrency data using the Yahoo Finance API and trains an LSTM model to make predictions.

## How to Use

1. Select the dataset for prediction: Choose from the available options, including Google (GOOG), Microsoft (MSFT), GameStop (GME), Infosys (INFY), Ethereum (ETH-USD), and Sound Financial Corporation (SBCF).

2. Explore Raw Data: The application displays the raw data, including open and close prices, in a table for the selected asset.

3. Visualize Time Series Data: A plot with open and close prices over time, equipped with a rangeslider for interactive exploration, is provided.

4. LSTM Model Training: The LSTM model is trained using historical closing prices. The model architecture includes two LSTM layers followed by a Dense layer. The training process is visualized with a progress bar.

5. Forecast Data: The application predicts future prices based on the trained LSTM model and displays the results alongside historical and actual data.

## Requirements

- Python 3.x
- Streamlit
- yfinance
- numpy
- pandas
- scikit-learn
- tensorflow
- plotly

Install the required dependencies using:
pip install -r requirements.txt


## How to Run
Run the application using the following command:
streamlit run app.py


