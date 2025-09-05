import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data_utils import generate_sine_wave
from transformer_model import TimeSeriesTransformer

# Hyperparameters
SEQ_LENGTH = 20
NUM_SAMPLES = 1000
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001

# Generate data
X, y = generate_sine_wave(SEQ_LENGTH, NUM_SAMPLES)
X_train, y_train = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]

# Convert to tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Model
model = TimeSeriesTransformer(input_size=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0
    for i in range(0, X_train.size(0), BATCH_SIZE):
        indices = permutation[i:i+BATCH_SIZE]
        batch_x, batch_y = X_train[indices], y_train[indices]
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/(X_train.size(0)//BATCH_SIZE):.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    preds = model(X_test)
    test_loss = criterion(preds, y_test).item()
    print(f"Test Loss: {test_loss:.4f}")

# Plot predictions
plt.figure(figsize=(10,4))
plt.plot(y_test.cpu().numpy(), label='True')
plt.plot(preds.cpu().numpy(), label='Predicted')
plt.legend()
plt.title('Time Series Forecasting with Transformer')
plt.show()
