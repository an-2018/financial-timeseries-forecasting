import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


def generate_temperature_series(n_dias=100, seed=0):
    np.random.seed(seed)
    dias = np.arange(n_dias)
    temperatura = 20 + 5 * np.sin(0.2 * dias) + np.random.normal(0, 0.5, size=n_dias)
    return pd.DataFrame({"dia": dias, "temperatura": temperatura})


def plot_normalization_demo(serie):
    dias = serie["dia"].values
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(dias, serie["temperatura"], label="Original")
    axes[0, 0].set_title("Temperatura Original")
    axes[0, 0].legend()

    scaler = MinMaxScaler(feature_range=(0, 1))
    norm = scaler.fit_transform(serie[["temperatura"]])
    axes[0, 1].plot(dias, norm, label="Min-Max [0,1]")
    axes[0, 1].set_title("Min-Max [0,1]")
    axes[0, 1].legend()

    scaler = MinMaxScaler(feature_range=(-1, 1))
    norm = scaler.fit_transform(serie[["temperatura"]])
    axes[1, 0].plot(dias, norm, label="Min-Max [-1,1]")
    axes[1, 0].set_title("Min-Max [-1,1]")
    axes[1, 0].legend()

    scaler = StandardScaler()
    norm = scaler.fit_transform(serie[["temperatura"]])
    axes[1, 1].plot(dias, norm, label="Z-score")
    axes[1, 1].set_title("Z-score (Standardization)")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def get_scalers():
    return {
        "MinMaxScaler": MinMaxScaler(),
        "StandardScaler": StandardScaler(),
        "RobustScaler": RobustScaler(),
    }


def normalize_dataframe(df, scaler):
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns)


def plot_normalizations(df, scalers, title_prefix=""):
    for nome, scaler in scalers.items():
        df_scaled = normalize_dataframe(df, scaler)
        plt.figure(figsize=(12, 5))
        for coluna in df_scaled.columns:
            plt.plot(df_scaled.index, df_scaled[coluna], label=coluna)
        plt.title(f"{title_prefix} - {nome}" if title_prefix else nome)
        plt.ylabel("Valor Normalizado")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
