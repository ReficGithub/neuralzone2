import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
import yfinance as yf
import plotly.graph_objects as go
import pytz


# Functie om financiële gegevens op te halen en voor te bereiden, vergelijkbaar met selecteer_kolommen
def bereid_financiële_gegevens_voor(financiële_gegevens):
    # Kopieer DataFrame om waarschuwingen te voorkomen
    financiële_gegevens = financiële_gegevens.copy()
    financiële_gegevens[['Open', 'High', 'Low', 'Close']] /= 10000
    financiële_gegevens['Weekdag'] = financiële_gegevens.index.weekday
    financiële_gegevens['Weekdag'] /= 10
    financiële_gegevens.fillna(0.0, inplace=True)
    financiële_gegevens.replace(0.0, 1e-7, inplace=True)
    financiële_gegevens.drop(['Adj Close', 'Volume'], axis=1, inplace=True)
    financiële_gegevens = financiële_gegevens.tail(30)
    return financiële_gegevens

def main():
    model_naam = "SPX1000x4.keras"
    model = load_model(model_naam)

    symbool = '^SPX'
    einddatum = datetime.now()
    amsterdam_tijdzone = pytz.timezone('Europe/Amsterdam')
    huidig_tijdstip_amsterdam = datetime.now(amsterdam_tijdzone)
    gewenst_tijdstip = amsterdam_tijdzone.localize(datetime(huidig_tijdstip_amsterdam.year, huidig_tijdstip_amsterdam.month, huidig_tijdstip_amsterdam.day, 22, 0))
    if huidig_tijdstip_amsterdam > gewenst_tijdstip:
        print("Het is later dan 22:00 in Amsterdam.")
        einddatum = einddatum + timedelta(days=1)
    einddatum = einddatum.strftime('%Y-%m-%d')
    sequence_length = 60

    einddatum_datetime = datetime.strptime(einddatum, '%Y-%m-%d')
    startdatum_datetime = einddatum_datetime - timedelta(days=sequence_length)
    startdatum = startdatum_datetime.strftime('%Y-%m-%d')

    financiële_gegevens = yf.download(symbool, start=startdatum, end=einddatum)
    financiële_gegevens = bereid_financiële_gegevens_voor(financiële_gegevens)

    invoer_reeks = financiële_gegevens.iloc[-sequence_length:].values
    voorspelling = model.predict(np.expand_dims(invoer_reeks, axis=0))
    voorspelling *= 10000
    invoer_reeks *= 10000
    # Afdrukken van de voorspelling
    print("Voorspelde sluitingsprijs voor de volgende dag:", voorspelling[0][1])

    # Plot de invoerreeks en de voorspellingskaars als candlesticks met Plotly

if __name__ == "__main__":
    main()