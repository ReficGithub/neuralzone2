# coding: iso-8859-1 -*-
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import load_model
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def bereid_financiële_gegevens_voor(financiële_gegevens):
    financiële_gegevens = financiële_gegevens.copy()
    financiële_gegevens[['Open', 'High', 'Low', 'Close']] /= 10000
    financiële_gegevens['Weekdag'] = financiële_gegevens.index.weekday
    financiële_gegevens['Weekdag'] /= 10
    financiële_gegevens.fillna(0.0, inplace=True)
    financiële_gegevens.replace(0.0, 1e-7, inplace=True)
    financiële_gegevens.drop(['Adj Close', 'Volume'], axis=1, inplace=True)
    return financiële_gegevens

def haal_financiële_gegevens_op(symbool, startdatum, einddatum):
    df = yf.download(symbool, start=startdatum, end=einddatum)
    return df

def voorbereid_validatiedataset(symbool, sequence_length, aantal_reeksen=1600):
    einddatum = datetime.now().strftime('%Y-%m-%d')  # Huidige datum
    startdatum_datetime = datetime.now() - timedelta(days=5*365)  # 15 jaar geleden
    startdatum = startdatum_datetime.strftime('%Y-%m-%d')

    financiële_gegevens = haal_financiële_gegevens_op(symbool, startdatum, einddatum)
    financiële_gegevens = bereid_financiële_gegevens_voor(financiële_gegevens)

    X, y = [], []
    for _ in range(aantal_reeksen):
        max_startpunt = len(financiële_gegevens) - (sequence_length + 1)
        startpunt = random.randint(0, max_startpunt)
        X.append(financiële_gegevens.iloc[startpunt:startpunt+sequence_length].values)
        y.append(financiële_gegevens.iloc[startpunt+sequence_length][['Open', 'Close']].values)  # Meerdere output waarden
    X = np.array(X)
    y = np.array(y)
    return X, y

def evalueer_model(model, X, y):
    voorspellingen = model.predict(X)
    voorspellingen*=10000
    y*=10000
    mae = mean_absolute_error(y, voorspellingen)
    mse = mean_squared_error(y, voorspellingen)
    
    return mae, mse, voorspellingen
def main():

    symbool = '^SPX'
    sequence_length = 30 
    X, y = voorbereid_validatiedataset(symbool, sequence_length)
    # print("Vorm van Invoer (X):", X.shape)
    # print("Vorm van Uitvoer (y):", y.shape)
    
    model_naam = 'SPX1000x4.keras'
    model = load_model(model_naam)
    
    mae, mse, voorspellingen = evalueer_model(model, X, y)
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")

if __name__ == "__main__":
    main()
