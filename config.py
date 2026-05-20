from dataclasses import dataclass, field


@dataclass
class Config:
    ticker: str = "AAPL"
    period: str = "5y"
    n_steps: int = 20
    n_dias_prever: int = 7
    lstm_units: int = 50
    lstm_layers: int = 1
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    random_seed: int = 42

    test_split: float = 0.15
    val_split: float = 0.15

    window_size: int = 10
    savgol_window: int = 21
    savgol_polyorder: int = 3
    fft_cutoff: float = 2.0

    use_simulated_data: bool = False
    sim_dias: int = 100
    sim_amplitude: float = 5.0
    sim_noise: float = 0.5

    features: tuple = field(default_factory=lambda: ("Open", "High", "Low", "Close"))


@dataclass
class GridConfig:
    n_steps: list = field(default_factory=lambda: [10, 20])
    lstm_units: list = field(default_factory=lambda: [50, 100])
    lstm_layers: list = field(default_factory=lambda: [1, 2])
    batch_size: list = field(default_factory=lambda: [32])
    learning_rate: list = field(default_factory=lambda: [0.001, 0.01])
