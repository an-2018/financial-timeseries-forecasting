import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def cria_sequencias(dataset, n_steps):
    X, y = [], []
    for i in range(n_steps, len(dataset)):
        X.append(dataset[i - n_steps : i])
        y.append(dataset[i])
    return np.array(X), np.array(y)


def build_lstm_model(n_steps, units=50):
    modelo = Sequential()
    modelo.add(LSTM(units, activation="relu", input_shape=(n_steps, 1)))
    modelo.add(Dense(1))
    modelo.compile(optimizer="adam", loss="mse")
    return modelo


def train_model(modelo, X, y, epochs=50):
    history = modelo.fit(X, y, epochs=epochs, verbose=0)
    return history


def plot_loss(history):
    plt.plot(history.history["loss"], label="Loss")
    plt.title("Evolução da Loss durante o Treino")
    plt.xlabel("Épocas")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.show()


def predict_future(modelo, last_sequence, scaler, n_dias_prever, n_steps):
    entrada = last_sequence.reshape(1, n_steps, 1)
    previsoes = []

    for _ in range(n_dias_prever):
        prox_valor = modelo.predict(entrada, verbose=0)[0][0]
        previsoes.append(prox_valor)
        nova_entrada = np.append(entrada[:, 1:, :], [[[prox_valor]]], axis=1)
        entrada = nova_entrada

    previsoes = scaler.inverse_transform(np.array(previsoes).reshape(-1, 1))
    return previsoes


def plot_prediction(fecho, previsoes, datas_prev):
    plt.figure(figsize=(10, 5))
    plt.plot(fecho.index, fecho.values, label="Histórico")
    plt.plot(
        datas_prev,
        previsoes,
        label="Previsão (próximos dias)",
        linestyle="--",
        marker="o",
    )
    plt.title("Previsão de Preços - Apple (LSTM)")
    plt.xlabel("Data")
    plt.ylabel("Preço ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
