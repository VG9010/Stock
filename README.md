# Stock
Stock Prediction Using Machine Learning


# Stock Market Predictor

## Overview
The **Stock Market Predictor** is a machine learning-based app built with **Streamlit**, **Keras**, and **Yahoo Finance API**. This app allows users to predict stock prices based on historical data using a pre-trained deep learning model. It also includes moving average calculations and visualizations to help analyze stock price trends.

## Features
- **Stock Data Visualization**: Displays historical stock prices along with moving averages (MA50, MA100, MA200).
- **Stock Price Prediction**: Predicts future stock prices using a pre-trained neural network model.
- **Interactive Charts**: Allows users to visualize stock prices against moving averages and the predicted prices.
- **Customizable Date Range**: Users can select custom start and end dates for stock data retrieval.

## Requirements
To run this project, you need the following dependencies:

- **Streamlit** for the web framework
- **Keras** for the pre-trained model
- **Yahoo Finance (yfinance)** for fetching historical stock data
- **Matplotlib** for visualizing stock prices and moving averages
- **Scikit-learn** for scaling the data
- **Pandas** for data manipulation
- **NumPy** for numerical operations

## Installation

### Step 1: Install Dependencies

You can install the required dependencies using `pip`. Run the following command:

```bash
pip install streamlit yfinance keras matplotlib scikit-learn pandas numpy

#Run the code 
streamlit run app.py
