# coding: iso-8859-1 -*-
import yfinance as yf

symbool = "^SPX"
startdatum = "2020-01-01"
einddatum = "2023-01-01"

def haal_financiele_gegevens_op(symbool, startdatum, einddatum):
    df = yf.download(symbool, start=startdatum, end=einddatum)
    return df

financiele_gegevens = haal_financiele_gegevens_op(symbool, startdatum, einddatum)
print(financiele_gegevens)
