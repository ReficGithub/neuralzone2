# coding: iso-8859-1 -*-
import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import random
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd


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

def genereer_trainingsdata(symbool, sequence_length, aantal_reeksen=1):
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

def laad_en_spel_model(model_naam, invoer):
    model = load_model(model_naam)
    # Voorspel met het geladen model
    voorspellingen = model.predict(invoer)
    return voorspellingen

def visualiseer_candlesticks(X, voorspellingen=None, y=None):
    fig = make_subplots(rows=len(X), cols=1, shared_xaxes=True, vertical_spacing=0.1)
    X*=10000
    voorspellingen*=10000
    y*=10000
    for i, (candlestick_data, voorspelling_data) in enumerate(zip(X, voorspellingen)):
        candlestick = go.Candlestick(
            x=np.arange(len(candlestick_data)),
            open=candlestick_data[:, 0],
            high=candlestick_data[:, 1],
            low=candlestick_data[:, 2],
            close=candlestick_data[:, 3],
            name=f'Candlesticks {i + 1}'
        )

        voorspelling_open = voorspelling_data[0]
        voorspelling_close = voorspelling_data[1]
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=0,
                y0=voorspelling_open,
                x1=len(candlestick_data) - 1,
                y1=voorspelling_open,
                line=dict(color="green", width=2),
                xref="x",
                yref="y",
            )
        )
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=0,
                y0=voorspelling_close,
                x1=len(candlestick_data) - 1,
                y1=voorspelling_close,
                line=dict(color="red", width=2),
                xref="x",
                yref="y",
            )
        )
            # Voeg stippen toe op basis van y[0] en y[1]
        fig.add_trace(go.Scatter(
            x=[len(candlestick_data) - 1],
            y=[y[i, 0]],
            mode="markers",
            marker=dict(color="green", size=10, symbol="circle"),
            name=f'Stip y[0] {i + 1}',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=[len(candlestick_data) - 1],
            y=[y[i, 1]],
            mode="markers",
            marker=dict(color="red", size=10, symbol="circle"),
            name=f'Stip y[1] {i + 1}',
            showlegend=False
        ))

        fig.add_trace(candlestick, row=i + 1, col=1)

    fig.update_layout(
        title='Candlestick-grafieken met Voorspellingen (Open en Close)',
        xaxis_title='Tijdstap',
        yaxis_title='Prijs',
        showlegend=True,
    )

    fig.show()

if __name__ == "__main__":
    symbool = '^SPX'
    sequence_length = 30
    modelnaam = "SPX1000x4.keras"
    X, y = genereer_trainingsdata(symbool, sequence_length)
    
    # Hier kun je X en y gebruiken voor verdere verwerking of visualisatie
    # Bijvoorbeeld:
    # print("Vorm van Invoer (X):", X.shape)
    # print("Vorm van Uitvoer (y):", y.shape)
    voorspellingen = laad_en_spel_model(modelnaam, X)
    visualiseer_candlesticks(X, voorspellingen, y)

