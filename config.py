from dataclasses import dataclass, field

@dataclass
class Config:
    ticker: str = "AAPL"
    period: str = "1y"
    n_steps: int = 20
    n_dias_prever: int = 7
    lstm_units: int = 50
    epochs: int = 50
    random_seed: int = 42
    
    # filtering
    window_size: int = 10
    savgol_window: int = 21
    savgol_polyorder: int = 3
    fft_cutoff: float = 2.0
    
    # simulation (for demo without yfinance)
    use_simulated_data: bool = False
    sim_dias: int = 100
    sim_amplitude: float = 5.0
    sim_noise: float = 0.5
    
    features: tuple = field(default_factory=lambda: ("Open", "High", "Low", "Close"))
