import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Configure the page
st.set_page_config(page_title="Stock Market Predictor", layout="wide", page_icon="📈")

# Load pre-trained model
model = load_model("C:/Users/Vaibhav/stock/Stock Predictions Model.keras")

# Sidebar for user input
st.sidebar.title("Stock Market Predictor")
st.sidebar.write("Predict and analyze stock prices with AI.")
stock = st.sidebar.text_input("Enter Stock Symbol", "GOOG")
start = st.sidebar.date_input("Start Date", value=pd.to_datetime("2002-01-01"))
end = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-11-11"))

# Page header
st.title("📊 Stock Market Predictor")
st.write("Enter a stock symbol to analyze historical data and predict future prices using AI.")

# Fetch stock data from Yahoo Finance
if stock:
    try:
        # Data retrieval
        data = yf.download(stock, start, end)
        if data.empty:
            st.error(f"No data found for stock symbol: {stock}")
        else:
            # Display stock data
            st.subheader(f"Stock Data for {stock}")
            st.dataframe(data.tail(10))

            # Split data into train and test
            data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
            data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            past_100_days = data_train.tail(100)
            data_test = pd.concat([past_100_days, data_test], ignore_index=True)
            data_test_scale = scaler.fit_transform(data_test)

            # Moving averages and plots
            st.subheader("📈 Moving Averages")
            col1, col2 = st.columns(2)

            with col1:
                st.write("### Price vs MA50")
                ma_50_days = data.Close.rolling(50).mean()
                fig1 = plt.figure(figsize=(8, 4))
                plt.plot(ma_50_days, "r", label="MA50")
                plt.plot(data.Close, "g", label="Price")
                plt.legend()
                st.pyplot(fig1)

            with col2:
                st.write("### Price vs MA50 vs MA100")
                ma_100_days = data.Close.rolling(100).mean()
                fig2 = plt.figure(figsize=(8, 4))
                plt.plot(ma_50_days, "r", label="MA50")
                plt.plot(ma_100_days, "b", label="MA100")
                plt.plot(data.Close, "g", label="Price")
                plt.legend()
                st.pyplot(fig2)
            
            # with col3:
            #     st.write("### Price vs MA100 vs MA200")
            #     ma_200_days = data.Close.rolling(200).mean()
            #     fig3 = plt.figure(figsize=(8, 4))
            #     plt.plot(ma_100_days, "r", label="MA100")
            #     plt.plot(ma_200_days, "b", label="MA200")
            #     plt.plot(data.Close, "g", label="Price")
            #     plt.legend()
            #     st.pyplot(fig3)  
               # Prediction data preparation
            x = []
            y = []
            for i in range(100, data_test_scale.shape[0]):
                x.append(data_test_scale[i - 100:i])
                y.append(data_test_scale[i, 0])
            x, y = np.array(x), np.array(y)

            # Predictions
            predict = model.predict(x)

            # Rescale predictions
            scale = 1 / scaler.scale_[0]
            predict = predict * scale
            y = y * scale

            # Predictions plot
            st.subheader("📉 Original Price vs Predicted Price")
            fig3 = plt.figure(figsize=(10, 5))
            plt.plot(predict, "r", label="Predicted Price")
            plt.plot(y, "g", label="Original Price")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.legend()
            st.pyplot(fig3)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please enter a stock symbol to proceed.")
