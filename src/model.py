import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


def cria_sequencias(dataset, n_steps):
    X, y = [], []
    for i in range(n_steps, len(dataset)):
        X.append(dataset[i - n_steps : i])
        y.append(dataset[i])
    return np.array(X), np.array(y)


def split_time_series(data, val_split=0.15, test_split=0.15):
    n = len(data)
    test_size = int(n * test_split)
    val_size = int(n * val_split)
    train_size = n - test_size - val_size
    train = data[:train_size]
    val = data[train_size : train_size + val_size]
    test = data[train_size + val_size :]
    return train, val, test


def prepare_sequences(train_raw, val_raw, test_raw, n_steps):
    X_train, y_train = cria_sequencias(train_raw, n_steps)

    if len(val_raw) > 0:
        val_context = np.concatenate([train_raw[-n_steps:], val_raw])
        X_val, y_val = cria_sequencias(val_context, n_steps)
    else:
        X_val, y_val = np.array([]), np.array([])

    if len(test_raw) > 0:
        test_context = np.concatenate([val_raw[-n_steps:], test_raw])
        X_test, y_test = cria_sequencias(test_context, n_steps)
    else:
        X_test, y_test = np.array([]), np.array([])

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_lstm_model(n_steps, units=50, n_layers=1, learning_rate=0.001):
    modelo = Sequential()
    modelo.add(
        LSTM(
            units,
            activation="relu",
            input_shape=(n_steps, 1),
            return_sequences=n_layers > 1,
        )
    )
    for i in range(1, n_layers):
        modelo.add(
            LSTM(units, activation="relu", return_sequences=i < n_layers - 1)
        )
    modelo.add(Dense(1))
    modelo.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return modelo


def train_model(modelo, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
    validation_data = (X_val, y_val) if X_val is not None and len(X_val) > 0 else None
    history = modelo.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        verbose=0,
    )
    return history


def plot_loss(history, titulo="Evolução da Loss durante o Treino"):
    plt.plot(history.history["loss"], label="Loss (treino)")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Loss (validação)")
    plt.title(titulo)
    plt.xlabel("Épocas")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.show()


def grid_search(raw_data, val_split, test_split, grid_cfg, epochs=30):
    train_raw, val_raw, test_raw = split_time_series(raw_data, val_split, test_split)

    best_val_loss = float("inf")
    best_model = None
    best_params = None
    best_history = None

    total = (
        len(grid_cfg.n_steps)
        * len(grid_cfg.lstm_units)
        * len(grid_cfg.lstm_layers)
        * len(grid_cfg.batch_size)
        * len(grid_cfg.learning_rate)
    )
    count = 0

    for n_steps in grid_cfg.n_steps:
        if n_steps >= len(train_raw):
            continue
        X_train, y_train, X_val, y_val, _, _ = prepare_sequences(
            train_raw, val_raw, test_raw, n_steps
        )
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        if len(X_val) > 0:
            X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

        for units in grid_cfg.lstm_units:
            for layers in grid_cfg.lstm_layers:
                for batch in grid_cfg.batch_size:
                    for lr in grid_cfg.learning_rate:
                        count += 1
                        print(f"  [{count}/{total}] n_steps={n_steps}, units={units}, layers={layers}, batch={batch}, lr={lr}")

                        modelo = build_lstm_model(n_steps, units, layers, lr)
                        history = train_model(
                            modelo, X_train, y_train, X_val, y_val,
                            epochs=epochs, batch_size=batch,
                        )

                        val_loss = min(history.history["val_loss"]) if len(X_val) > 0 else history.history["loss"][-1]

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_model = modelo
                            best_params = {
                                "n_steps": n_steps,
                                "units": units,
                                "layers": layers,
                                "batch_size": batch,
                                "learning_rate": lr,
                            }
                            best_history = history

    print(f"\n  Melhor val_loss: {best_val_loss:.6f}")
    print(f"  Melhores parâmetros: {best_params}")
    return best_model, best_params, best_history


def evaluate_model(modelo, X_test, y_test, scaler):
    y_pred_scaled = modelo.predict(X_test, verbose=0)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def plot_test_prediction(modelo, X_test, y_test, scaler, datas, n_steps):
    y_pred_scaled = modelo.predict(X_test, verbose=0)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    plt.figure(figsize=(12, 5))
    plt.plot(datas, y_true, label="Real (Teste)", color="blue", marker=".", linestyle="-")
    plt.plot(datas, y_pred, label="Previsto (1-step)", color="red", marker="x", linestyle="--")
    plt.title(f"Previsão 1-Step no Conjunto de Teste — RMSE={rmse:.2f}, MAE={mae:.2f}")
    plt.xlabel("Data")
    plt.ylabel("Preço ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_extended_forecast(fecho, train_raw, val_raw, test_raw, model, scaler, n_steps, n_prever):
    seed = np.concatenate([train_raw, val_raw])[-n_steps:]
    entrada = seed.reshape(1, n_steps, 1)
    previsoes_scaled = []
    for _ in range(min(n_prever, len(test_raw))):
        prox_valor = model.predict(entrada, verbose=0)[0][0]
        previsoes_scaled.append(prox_valor)
        entrada = np.append(entrada[:, 1:, :], [[[prox_valor]]], axis=1)

    pred = scaler.inverse_transform(np.array(previsoes_scaled).reshape(-1, 1)).flatten()

    actual = scaler.inverse_transform(test_raw[:n_prever].reshape(-1, 1)).flatten()

    n_train = len(train_raw)
    n_val = len(val_raw)
    n_test = len(test_raw)

    plt.figure(figsize=(14, 6))

    plt.plot(
        fecho.index[:n_train],
        fecho.values[:n_train],
        label="Treino", color="blue",
    )

    if n_val > 0:
        plt.plot(
            fecho.index[n_train : n_train + n_val],
            fecho.values[n_train : n_train + n_val],
            label="Validação", color="green", alpha=0.7,
        )

    n_plot = min(n_prever, n_test)
    datas_test = fecho.index[n_train + n_val : n_train + n_val + n_plot]

    plt.plot(datas_test, actual, label="Real (Teste)", color="orange", marker=".", linewidth=2)
    plt.plot(datas_test, pred, label=f"Previsão Recursiva ({n_plot} dias)", color="red", linestyle="--", marker="x")

    plt.title(f"Previsão Multi-Passo no Teste — {n_plot} dias")
    plt.xlabel("Data")
    plt.ylabel("Preço ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
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
    plt.title("Previsão de Preços Futuros (LSTM)")
    plt.xlabel("Data")
    plt.ylabel("Preço ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
