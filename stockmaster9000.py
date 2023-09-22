import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
import yfinance as yf
import plotly.graph_objects as go
import pytz


# Functie om financiŽle gegevens op te halen en voor te bereiden, vergelijkbaar met selecteer_kolommen
def bereid_financiŽle_gegevens_voor(financiŽle_gegevens):
    # Kopieer DataFrame om waarschuwingen te voorkomen
    financiŽle_gegevens = financiŽle_gegevens.copy()
    financiŽle_gegevens[['Open', 'High', 'Low', 'Close']] /= 10000
    financiŽle_gegevens['Weekdag'] = financiŽle_gegevens.index.weekday
    financiŽle_gegevens['Weekdag'] /= 10
    financiŽle_gegevens.fillna(0.0, inplace=True)
    financiŽle_gegevens.replace(0.0, 1e-7, inplace=True)
    financiŽle_gegevens.drop(['Adj Close', 'Volume'], axis=1, inplace=True)
    financiŽle_gegevens = financiŽle_gegevens.tail(30)
    return financiŽle_gegevens

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

    financiŽle_gegevens = yf.download(symbool, start=startdatum, end=einddatum)
    financiŽle_gegevens = bereid_financiŽle_gegevens_voor(financiŽle_gegevens)

    invoer_reeks = financiŽle_gegevens.iloc[-sequence_length:].values
    voorspelling = model.predict(np.expand_dims(invoer_reeks, axis=0))
    voorspelling *= 10000
    invoer_reeks *= 10000
    # Afdrukken van de voorspelling
    print("Voorspelde sluitingsprijs voor de volgende dag:", voorspelling[0][1])

    # Plot de invoerreeks en de voorspellingskaars als candlesticks met Plotly

if __name__ == "__main__":
    main()