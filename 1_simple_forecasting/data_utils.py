import numpy as np

def generate_sine_wave(seq_length, num_samples):
    X = []
    y = []
    for _ in range(num_samples):
        freq = np.random.uniform(0.1, 0.5)
        phase = np.random.uniform(0, np.pi)
        x = np.arange(seq_length + 1)
        series = np.sin(freq * x + phase)
        X.append(series[:-1])
        y.append(series[-1])
    return np.array(X), np.array(y)
