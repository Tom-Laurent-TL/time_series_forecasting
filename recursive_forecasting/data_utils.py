import numpy as np


def generate_sine_wave(num_points):
    """
    Generate a single sine wave signal of length num_points.
    Frequency and phase are randomized.
    Returns:
        signal: np.ndarray, shape (num_points,)
    """
    freq = np.random.uniform(0.1, 0.5)
    phase = np.random.uniform(0, np.pi)
    x = np.arange(num_points)
    signal = np.sin(freq * x + phase)
    return signal

def generate_complex_signal(num_points):
    """
    Generate a complex signal by combining multiple sine waves and adding noise.
    Returns:
        signal: np.ndarray, shape (num_points,)
    """
    x = np.arange(num_points)
    # Random frequencies and phases for three sine components
    freq1 = np.random.uniform(0.05, 0.2)
    freq2 = np.random.uniform(0.2, 0.5)
    freq3 = np.random.uniform(0.5, 1.0)
    phase1 = np.random.uniform(0, np.pi)
    phase2 = np.random.uniform(0, np.pi)
    phase3 = np.random.uniform(0, np.pi)
    # Amplitudes
    amp1 = np.random.uniform(0.5, 1.0)
    amp2 = np.random.uniform(0.2, 0.7)
    amp3 = np.random.uniform(0.1, 0.5)
    # Combine sine waves
    signal = (amp1 * np.sin(freq1 * x + phase1) +
                amp2 * np.sin(freq2 * x + phase2) +
                amp3 * np.sin(freq3 * x + phase3))

    # Multi-scale trends
    # Slow linear trend
    linear_trend = np.random.uniform(-0.01, 0.01) * x
    # Quadratic trend
    quadratic_trend = np.random.uniform(-0.0001, 0.0001) * (x ** 2)
    # Very low-frequency sinusoidal trend
    slow_freq = np.random.uniform(0.005, 0.02)
    slow_phase = np.random.uniform(0, np.pi)
    slow_amp = np.random.uniform(0.5, 1.5)
    slow_trend = slow_amp * np.sin(slow_freq * x + slow_phase)

    signal += linear_trend + quadratic_trend + slow_trend

    # Add Gaussian noise
    noise = np.random.normal(0, 0.1, num_points)
    signal += noise
    return signal
