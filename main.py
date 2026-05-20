from config import Config
from src.filtering import generate_signal, plot_filtering_demo
from src.normalization import (
    generate_temperature_series,
    plot_normalization_demo,
    plot_normalizations,
    get_scalers,
)
from src.data_loader import load_data, plot_close, plot_ohlc
from src.model import (
    cria_sequencias,
    build_lstm_model,
    train_model,
    plot_loss,
    predict_future,
    plot_prediction,
)
import pandas as pd
import matplotlib.pyplot as plt


def run_filtering_demo(cfg: Config):
    print("=== 2.1 — Filtragem dos Dados ===")
    t, sinal_limpo, sinal_ruidoso = generate_signal(seed=cfg.random_seed)
    plot_filtering_demo(
        t,
        sinal_limpo,
        sinal_ruidoso,
        window_size=cfg.window_size,
        savgol_window=cfg.savgol_window,
        savgol_polyorder=cfg.savgol_polyorder,
        fft_cutoff=cfg.fft_cutoff,
    )


def run_normalization_demo(cfg: Config):
    print("=== 2.2 — Normalização dos Dados ===")
    serie = generate_temperature_series(n_dias=cfg.sim_dias)
    plot_normalization_demo(serie)


def run_data_loading_demo(cfg: Config):
    print("=== 2.3 — Dados Financeiros ===")
    dados = load_data(ticker=cfg.ticker, period=cfg.period)
    plot_close(dados, ticker=cfg.ticker)
    plot_ohlc(dados, features=cfg.features)

    scalers = get_scalers()
    plot_normalizations(dados, scalers, title_prefix=cfg.ticker)

    return dados


def run_lstm_pipeline(cfg: Config, dados=None):
    print("=== 2.4 — LSTM para Previsão Financeira ===")

    if dados is None:
        dados = load_data(ticker=cfg.ticker, period=cfg.period)

    fecho = dados[["Close"]]

    # Normalizar
    scaler = MinMaxScaler()
    fecho_scaled = scaler.fit_transform(fecho)

    # Criar sequências
    X, y = cria_sequencias(fecho_scaled, cfg.n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Construir e treinar modelo
    modelo = build_lstm_model(cfg.n_steps, units=cfg.lstm_units)
    history = train_model(modelo, X, y, epochs=cfg.epochs)

    plot_loss(history)

    # Prever
    last_seq = fecho_scaled[-cfg.n_steps :]
    previsoes = predict_future(
        modelo, last_seq, scaler, cfg.n_dias_prever, cfg.n_steps
    )

    # Criar índice temporal
    ult_data = fecho.index[-1]
    datas_prev = pd.date_range(
        start=ult_data + pd.Timedelta(days=1),
        periods=cfg.n_dias_prever,
        freq="B",
    )

    plot_prediction(fecho, previsoes, datas_prev)


def main():
    cfg = Config()

    run_filtering_demo(cfg)
    run_normalization_demo(cfg)
    dados = run_data_loading_demo(cfg)
    run_lstm_pipeline(cfg, dados=dados)


if __name__ == "__main__":
    main()
