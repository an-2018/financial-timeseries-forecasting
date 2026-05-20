from config import Config, GridConfig
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
    split_time_series,
    prepare_sequences,
    build_lstm_model,
    train_model,
    plot_loss,
    grid_search,
    evaluate_model,
    plot_test_prediction,
    plot_extended_forecast,
    predict_future,
    plot_prediction,
)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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


def run_lstm_pipeline(cfg: Config, grid_cfg: GridConfig, dados=None):
    print("=== 2.4 — LSTM para Previsão Financeira ===")

    if dados is None:
        dados = load_data(ticker=cfg.ticker, period=cfg.period)

    print(f"Dados carregados: {len(dados)} dias")
    fecho = dados[["Close"]]

    scaler = MinMaxScaler()
    fecho_scaled = scaler.fit_transform(fecho)

    train_raw, val_raw, test_raw = split_time_series(
        fecho_scaled, cfg.val_split, cfg.test_split
    )
    print(
        f"Split: {len(train_raw)} treino / {len(val_raw)} val / {len(test_raw)} teste"
    )

    print("\nA iniciar grid search de hiperparâmetros...")
    best_model, best_params, best_history = grid_search(
        fecho_scaled, cfg.val_split, cfg.test_split, grid_cfg, cfg.epochs
    )

    plot_loss(best_history, "Grid Search — Melhor Modelo (Loss)")

    _, _, _, _, X_test, y_test = prepare_sequences(
        train_raw, val_raw, test_raw, best_params["n_steps"]
    )
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    metrics = evaluate_model(best_model, X_test, y_test, scaler)
    print("\n=== Métricas no Conjunto de Teste (1-step ahead) ===")
    for nome, valor in metrics.items():
        print(f"  {nome}: {valor:.4f}")

    datas_teste = fecho.index[-len(test_raw) + best_params["n_steps"] :]

    plot_test_prediction(
        best_model, X_test, y_test, scaler, datas_teste, best_params["n_steps"]
    )

    N_PREVER_TESTE = 30
    plot_extended_forecast(
        fecho, train_raw, val_raw, test_raw,
        best_model, scaler, best_params["n_steps"], N_PREVER_TESTE,
    )

    dados_completos = np.concatenate([train_raw, val_raw, test_raw])
    last_seq = dados_completos[-best_params["n_steps"] :]
    previsoes = predict_future(
        best_model, last_seq, scaler, cfg.n_dias_prever, best_params["n_steps"]
    )

    ult_data = fecho.index[-1]
    datas_prev = pd.date_range(
        start=ult_data + pd.Timedelta(days=1),
        periods=cfg.n_dias_prever,
        freq="B",
    )

    plot_prediction(fecho, previsoes, datas_prev)


def main():
    cfg = Config()
    grid_cfg = GridConfig()

    run_filtering_demo(cfg)
    run_normalization_demo(cfg)
    dados = run_data_loading_demo(cfg)
    run_lstm_pipeline(cfg, grid_cfg, dados=dados)


if __name__ == "__main__":
    main()
