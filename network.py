# coding: iso-8859-1 -*-
import yfinance as yf
import numpy as np
import random
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense

def haal_financiele_gegevens_op(symbool, startdatum, einddatum):
    df = yf.download(symbool, start=startdatum, end=einddatum)
    print(df)
    return df

def voorbereid_trainingsdataset(symbool, einddatum, sequence_length, aantal_reeksen=6400):
    einddatum_datetime = datetime.strptime(einddatum, '%Y-%m-%d')
    startdatum_datetime = einddatum_datetime - timedelta(days=365*5)
    startdatum = startdatum_datetime.strftime('%Y-%m-%d')
    financiele_gegevens = haal_financiele_gegevens_op(symbool, startdatum, einddatum)
    kolommen = ['Open', 'High', 'Low', 'Close']
    financiele_gegevens[kolommen] /= 10000
    financiele_gegevens['Weekdag'] = financiele_gegevens.index.weekday / 10
    financiele_gegevens.fillna(0.0, inplace=True)
    financiele_gegevens.replace(0.0, 1e-7, inplace=True)
    financiele_gegevens.drop(['Adj Close', 'Volume'], axis=1, inplace=True)
    
    X, y = [], []
    
    for _ in range(aantal_reeksen):
        max_startpunt = len(financiele_gegevens) - (sequence_length + 1)
        startpunt = random.randint(0, max_startpunt)
        X.append(financiele_gegevens.iloc[startpunt:startpunt+sequence_length].values)
        y.append(financiele_gegevens.iloc[startpunt+sequence_length][["Open", "Close"]].values)

    X = np.array(X)
    y = np.array(y)
    return X, y

def bouw_lstm_netwerk(input_shape, output_dim):
    model = Sequential()
    model.add(LSTM(1000, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(1000, return_sequences=True))
    model.add(LSTM(1000, return_sequences=True))
    model.add(LSTM(1000))   
    model.add(Dense(output_dim))
    return model

def sla_model_op(model, model_naam):
    model.save(f'{model_naam}.keras')
    print(f'{model_naam}.keras is succesvol opgeslagen.')

symbools = ["^SPX"]
einddatum = '2023-08-08'
sequence_length = 30
batch_size = 32
epochs = 200

model_placeholder = None
bestaand_model = model_placeholder
X, y = voorbereid_trainingsdataset(symbools[0], einddatum, sequence_length)
if bestaand_model is None:
    input_shape = (X.shape[1], X.shape[2])
    output_dim = 2
    model = bouw_lstm_netwerk(input_shape, output_dim)
else:
    model = load_model(bestaand_model)

model.compile(loss='mean_absolute_percentage_error', optimizer='adam')

for symbool in symbools:
    X, y = voorbereid_trainingsdataset(symbool, einddatum, sequence_length)
    model.fit(X, y, epochs=epochs, batch_size=batch_size)

sla_model_op(model, "SPX1000x4")
