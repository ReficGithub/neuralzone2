# coding: iso-8859-1 -*-
import yfinance as yf

symbool = "^SPX"
startdatum = 2020-01-01
einddatum = 2023-01-01

financiele_gegevens = haal_financiele_gegevens_op(symbool, startdatum, einddatum)
print(financiele_gegevens)
