import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


def load_data(ticker="AAPL", period="6mo"):
    dados = yf.download(ticker, period=period)
    return dados


def plot_close(dados, ticker="AAPL"):
    dados["Close"].plot(
        title=f"Preço de Fecho - {ticker}", figsize=(10, 4), grid=True
    )
    plt.ylabel("USD")
    plt.show()


def plot_ohlc(dados, features=("Open", "High", "Low", "Close")):
    valores = dados[list(features)]
    plt.figure(figsize=(12, 6))
    for coluna in valores.columns:
        plt.plot(valores.index, valores[coluna], label=coluna)
    plt.title("AAPL - Abertura, Máximo, Mínimo e Fecho")
    plt.xlabel("Data")
    plt.ylabel("Preço (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
