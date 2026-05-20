import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.fftpack import fft, ifft


def generate_signal(n=500, seed=42, noise_level=0.4):
    np.random.seed(seed)
    t = np.linspace(0, 10, n)
    sinal_limpo = np.sin(2 * np.pi * 0.5 * t) + 0.5 * t
    ruido = np.random.normal(0, noise_level, size=t.shape)
    sinal_ruidoso = sinal_limpo + ruido
    return t, sinal_limpo, sinal_ruidoso


def moving_average(signal, window_size=10):
    return pd.Series(signal).rolling(window=window_size, center=True).mean()


def savgol(signal, window_length=21, polyorder=3):
    return savgol_filter(signal, window_length, polyorder)


def fft_lowpass(signal, t, cutoff=2.0):
    fft_sinal = fft(signal)
    frequencias = np.fft.fftfreq(len(t), d=t[1] - t[0])
    fft_filtrado = fft_sinal.copy()
    fft_filtrado[np.abs(frequencias) > cutoff] = 0
    return np.real(ifft(fft_filtrado))


def plot_filtering_demo(t, sinal_limpo, sinal_ruidoso, window_size=10,
                        savgol_window=21, savgol_polyorder=3, fft_cutoff=2.0):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(t, sinal_ruidoso, label="Sinal com ruído", color="gray")
    ax.plot(t, sinal_limpo, label="Sinal original (limpo)", color="black", linestyle="--")
    ax.set_title("Sinal Original e com Ruído")
    ax.legend()

    ax = axes[0, 1]
    sinal_movel = moving_average(sinal_ruidoso, window_size)
    ax.plot(t, sinal_movel, label="Média Móvel", color="blue")
    ax.set_title("Média Móvel")
    ax.legend()

    ax = axes[1, 0]
    sinal_savgol = savgol(sinal_ruidoso, savgol_window, savgol_polyorder)
    ax.plot(t, sinal_savgol, label="Savitzky-Golay", color="green")
    ax.set_title("Filtro de Savitzky-Golay")
    ax.legend()

    ax = axes[1, 1]
    sinal_fft = fft_lowpass(sinal_ruidoso, t, fft_cutoff)
    ax.plot(t, sinal_fft, label="FFT passa-baixo", color="orange")
    ax.set_title("Filtro FFT (passa-baixo)")
    ax.legend()

    plt.tight_layout()
    plt.show()
